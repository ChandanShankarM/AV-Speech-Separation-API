import abc
from functools import partial
import torch
from pydub import AudioSegment
#import ffmpeg


class AudioReader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, filename):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def length(audio) -> int:
        """
        Return length of audio in milliseconds
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def slice(audio, offset: int, duration: int):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def overlay(first, second):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def change_volume(audio, by_db):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def to_tensor(audio, channels_first: bool = True, normalization=None):
        raise NotImplementedError


class PydubAudioReader(AudioReader):
    def __init__(self, default_format, sr):
        self.default_format = default_format
        self.sr = sr

    def load(self, filename, format=None) -> AudioSegment:
        if not format:
            format = self.default_format
        return AudioSegment.from_file(filename, format).set_frame_rate(frame_rate=self.sr).set_channels(channels=1)

    @staticmethod
    def length(audio: AudioSegment) -> int:
        return len(audio)

    @staticmethod
    def slice(audio: AudioSegment, offset: int, duration: int):
        return audio[offset:offset + duration]

    @staticmethod
    def overlay(first: AudioSegment, second: AudioSegment):
        return first.overlay(second)

    @staticmethod
    def change_volume(audio, by_db):
        return audio + by_db

    @staticmethod
    def to_tensor(audio: AudioSegment, channels_first: bool = True, normalization=1 << 15):
        tensor = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float)
        tensor = tensor.view((audio.channels, -1) if channels_first else (-1, audio.channels))
        if normalization:
            # magic constant from https://stackoverflow.com/a/53633178
            tensor /= normalization
        return tensor


class PytorchAudioReader(AudioReader):
    pass


target_sr = 8000
_default_audio_reader = partial(PydubAudioReader, default_format="mp4", sr=8000)


def get_default_audio_reader():
    assert _default_audio_reader
    return _default_audio_reader()