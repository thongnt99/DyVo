import pickle
from typing import Iterator, Dict, Any, Callable
import numpy as np
import json
import lz4.frame
from tqdm import tqdm


def encode_lz4json(payload: Any) -> bytes:
    encoded_payload = json.dumps(payload).encode('utf8')
    encoded_payload = lz4.frame.compress(encoded_payload)
    return encoded_payload


def encode_json(payload: Any) -> bytes:
    encoded_payload = json.dumps(payload).encode('utf8')
    return encoded_payload


def encode_pickle(payload: Any) -> bytes:
    encoded_payload = pickle.dumps(payload)
    return encoded_payload


def decode_pickle(encoded_payload: bytes) -> Any:
    payload = pickle.loads(encoded_payload)
    return payload


def decode_json(encoded_payload: bytes) -> Any:
    payload = json.loads(encoded_payload)
    return payload


def decode_lz4json(encoded_payload: bytes) -> Any:
    encoded_payload = lz4.frame.decompress(encoded_payload)
    payload = json.loads(encoded_payload)
    return payload


class PayloadLookup:
    @staticmethod
    def build(path: str, record_iter: Iterator[Dict[str, Any]], encode: Callable[[Any], bytes], decode: Callable[[bytes], Any]) -> 'PayloadLookup':
        str_keys = {}
        with open(path+'.offsets.u64', 'wb') as f_offsets, \
                open(path+'.payloads.bin', 'wb') as f_payloads:
            # write initial offset
            f_offsets.write(np.array(f_payloads.tell(),
                            dtype=np.uint64).tobytes())
            for i, record in enumerate(tqdm(record_iter)):
                str_keys[record[0]] = i
                encoded_payload = encode(record[1])
                f_payloads.write(encoded_payload)
                f_payloads.flush()
                f_offsets.write(np.array(f_payloads.tell(),
                                dtype=np.uint64).tobytes())
        lookup_path = path + ".keylookup.json"
        json.dump(str_keys, open(lookup_path, "w"))
        return PayloadLookup(path, decode)

    def __contains__(self, item):
        """
        Override the behavior of the 'in' operator for instances of this class.

        Parameters:
        - item: The item to check for membership.

        Returns:
        - True if 'item' is in the collection, False otherwise.
        """
        return item in self.key_lookup

    def __init__(self, path: str, decode: Callable[[bytes], Any]):
        self.offsets = np.memmap(path + '.offsets.u64', dtype=np.uint64)
        self.f_payload = open(path + '.payloads.bin', 'rb')
        self.key_lookup = json.load(open(path + ".keylookup.json", "r"))
        self.decode = decode

    def __getitem__(self, doc_id: str) -> Any:
        doc_idx = self.key_lookup[doc_id]
        start, end = self.offsets[doc_idx:doc_idx+2]
        payload_len = end - start
        self.f_payload.seek(start)
        encoded_payload = self.f_payload.read(payload_len)
        return self.decode(encoded_payload)


if __name__ == '__main__':
    # test it out
    def sample_iter():
        yield 'hello', [1, 2, 3, 4]
        yield 'world', [2, 3, 4, 5]
        # yield {'doc_id': '0', 'payload': ['some', 'array', 'of', 'items']}
        # yield {'doc_id': '1', 'payload': []}
        # yield {'doc_id': '2', 'payload': ['another', 'array', 'of', 'items']}
        # yield {'doc_id': '3', 'payload': [1, 2, 3, "mixed types"]}
        # yield {'doc_id': '4', 'payload': {'dicts': 'too'}}

    lookup = PayloadLookup.build(
        'deleteme', sample_iter(), encode_lz4json, decode_lz4json)
    assert lookup['hello'] == [1, 2, 3, 4]
    assert lookup['world'] == [2, 3, 4, 5]
