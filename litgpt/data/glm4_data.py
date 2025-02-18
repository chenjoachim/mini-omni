# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from litgpt import PromptStyle
from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.tokenizer import Tokenizer

import whisper

text_vocabsize = 32000
text_specialtokens = 64
audio_vocabsize = 16384
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4

def layershift(input_id, layer, stride=16448, shift=32064):
    return input_id + shift + layer * stride

def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1

def get_input_ids_whisper(
    mel, leng, whispermodel, device, 
    special_token_a=_answer_a, special_token_t=_answer_t,
):

    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]

    T = audio_feature.size(0)
    input_ids = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return audio_feature.unsqueeze(0), input_ids


class GLM4Dataset(SFTDataset):
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        example = self.data[idx]
        # mel, leng = example['audio_path']
        
        # prompt = self.prompt_style.apply(prompt=example["input_text"], **example)
        # prompt_and_response = prompt + example["label_text"]
        # encoded_prompt = self.tokenizer.encode(prompt, max_length=self.max_seq_length)
        # encoded_prompt_and_response = self.tokenizer.encode(
        #     prompt_and_response, eos=True, max_length=self.max_seq_length
        # )

        # The labels are the full prompt with response, but with the prompt masked out
        # labels = encoded_prompt_and_response.clone()
        # if self.mask_prompt:
        #     labels[: len(encoded_prompt)] = self.ignore_index

        # return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64)}

        # Random audio features with typical Whisper dimensions
        audio_features = torch.randn(100, self.audio_dim)  # [T=100, audio_dim]
        
        T = audio_features.size(0)
        
        # Create 8 random input_id tensors
        input_ids = []
        for _ in range(8):
            # Random tokens between 0-999, keeping batch dim of 1
            tensor = torch.randint(0, 1000, (1, T+3))
            input_ids.append(tensor)
        
        # Optional: random labels
        labels = input_ids
        
        return {
            "audio_features": audio_features,
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class GLM4Data(DataModule):
    """Loads JSON or JSONL data for supervised finetuning."""

    json_path: Path
    """A path to a JSON file or a directory with `train.json` and `val.json` containing the data.
    The file(s) should contain a list of samples (dicts). Each dict must have the keys 'instruction' and 'output',
    and can optionally have a key 'input' (see Alpaca)."""
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: Optional[float] = None
    """The fraction of the dataset to use for the validation dataset. The rest is used for training.
    Only applies if you passed in a single file to `json_path`."""
    prompt_style: Union[str, PromptStyle] = "llama2"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.json_path.is_file() and self.val_split_fraction is None:
            raise ValueError(
                "If `json_path` is a file, you must set `val_split_fraction` to a value between 0 and 1 to split the"
                " data into train and validation sets."
            )
        if self.json_path.is_dir() and self.val_split_fraction is not None:
            raise ValueError(
                "If `json_path` is a directory, it must contain 'train.json' and 'val.json' files and"
                f" hence `val_split_fraction` should not be set. Got `{self.val_split_fraction=}`."
            )
        if not self.json_path.exists():
            raise FileNotFoundError(
                "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files,"
                f" but '{self.json_path!s}' does not exist."
            )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        train_data, test_data = self.get_splits()

        self.train_dataset = GLM4Dataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = GLM4Dataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def get_splits(self) -> Tuple:
        # A single file (gets split into train and test)
        if self.json_path.is_file():
            data = load_split(self.json_path)

            # Partition the dataset into train and test
            train_data, test_data = random_split(
                data,
                [1.0 - self.val_split_fraction, self.val_split_fraction],
                generator=torch.Generator().manual_seed(self.seed),
            )
            return train_data, test_data

        # A directory containing train.json and val.json
        if (train_file := self.find_split("train")) and (val_file := self.find_split("val")):
            train_data = load_split(train_file)
            test_data = load_split(val_file)
            return train_data, test_data

        raise FileNotFoundError(
            "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
        )

    def find_split(self, split_name: str) -> Optional[Path]:
        for suffix in (".json", ".jsonl"):
            if (file := self.json_path / f"{split_name}{suffix}").is_file():
                return file
        return None


def load_split(json_path: Path) -> Any:
    if json_path.suffix == ".json":
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    if json_path.suffix == ".jsonl":
        with open(json_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(f"Unsupported file format: {json_path.suffix}. Expected `.json` or `.jsonl`.")