import torch
import gzip


class STRINGDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        links_path,
        seqs_path,
        global_rank=0,
        world_size=1,
        concat=False,
        max_examples=None,
        max_len=None,
        overfit=False,
        seek=None,
    ):
        super().__init__()
        self.links_path = links_path
        self.seqs_path = seqs_path
        self.global_rank = global_rank
        self.world_size = world_size

        if max_examples:
            self.max_iters = int(max_examples // world_size)
        else:
            self.max_iters = None
        self.concat = concat
        self.max_len = max_len
        self.overfit = overfit
        self.seek = seek

    def __len__(self):
        return self.max_iters

    def __iter__(self):
        it = self.__iter_helper__()
        for i, n in enumerate(it):
            if self.seek and i < self.seek:
                pass
            else:
                yield n

    def __iter_helper__(self):
        self.seqs = {}
        links_f = iter(gzip.open(self.links_path, "rt"))
        seqs_f = iter(gzip.open(self.seqs_path, "rt"))
        i, j = 0, 0
        while True:
            try:
                name1, name2 = next(links_f).strip().split()[:2]
                if name1 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq
                if name2 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq
                if self.max_len and not (
                    len(self.seqs[name1]) <= self.max_len and len(self.seqs[name2]) <= self.max_len
                ):
                    continue  # don't increment i
                if i % self.world_size == self.global_rank:
                    if self.concat:
                        if self.overfit:
                            while True:
                                yield (self.seqs[name1] + "G" * 25 + self.seqs[name2],)
                        else:
                            yield (self.seqs[name1] + "G" * 25 + self.seqs[name2],)
                    else:
                        if self.overfit:
                            while True:
                                yield self.seqs[name1], self.seqs[name2]
                        else:
                            yield self.seqs[name1], self.seqs[name2]
                    j += 1
                i += 1
                if j == self.max_iters:
                    break
            except StopIteration:
                links_f = iter(gzip.open(self.links_path, "rt"))
