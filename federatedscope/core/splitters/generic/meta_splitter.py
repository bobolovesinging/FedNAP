from numbers import Integral
import numpy as np
import logging

from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.generic import IIDSplitter

logger = logging.getLogger(__name__)


class MetaSplitter(BaseSplitter):
    """
    This splitter split dataset with meta information with LLM dataset.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
    """
    def __init__(self,
                 client_num,
                 on_insufficient='error',
                 preserve_category_ids=False,
                 unknown_category='error',
                 **kwargs):
        super(MetaSplitter, self).__init__(client_num)
        if on_insufficient not in ('error', 'iid'):
            raise ValueError('on_insufficient must be "error" or "iid"')
        if unknown_category not in ('error', 'server_only'):
            raise ValueError(
                'unknown_category must be "error" or "server_only"')
        self.on_insufficient = on_insufficient
        self.preserve_category_ids = preserve_category_ids
        self.unknown_category = unknown_category
        self.iid_spliter = IIDSplitter(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        if getattr(dataset, 'replicate_to_clients', False):
            logger.info(
                'Replicating a fixed public evaluation panel to %d clients.',
                self.client_num)
            return [dataset for _ in range(self.client_num)]

        tmp_dataset = [ds for ds in dataset]
        if isinstance(tmp_dataset[0], tuple):
            label = np.array([y for x, y in tmp_dataset])
        elif isinstance(tmp_dataset[0], dict):
            label = np.array([x['categories'] for x in tmp_dataset])
        else:
            raise TypeError(f'Unsupported data formats {type(tmp_dataset[0])}')

        if self.preserve_category_ids:
            idx_slice = [[] for _ in range(self.client_num)]
            unknown = []
            for idx, category in enumerate(label):
                if isinstance(category, Integral) and \
                        0 <= int(category) < self.client_num:
                    idx_slice[int(category)].append(idx)
                else:
                    unknown.append((idx, category))
            if unknown and self.unknown_category == 'error':
                values = sorted({repr(category) for _, category in unknown})
                raise ValueError(
                    'MetaSplitter received category IDs outside the client '
                    f'range [0, {self.client_num - 1}]: {values[:10]}')
            if unknown:
                logger.info(
                    'Keeping %d samples from unseen categories on the '
                    'server-only dataset.', len(unknown))
            return self._build_splits(dataset, idx_slice)

        # Use a stable category order so repeated train/val/test calls do not
        # silently permute client ownership.
        categories = sorted(set(label), key=lambda value: repr(value))
        idx_slice = []
        for cat in categories:
            idx_slice.append(np.where(np.array(label) == cat)[0].tolist())

        # print the size of each categories
        tot_size = 0
        for i, cat in enumerate(categories):
            logger.info(f'Index: {i}\t'
                        f'Category: {cat}\t'
                        f'Size: {len(idx_slice[i])}')
            tot_size += len(idx_slice[i])
        logger.info(f'Total size: {tot_size}')

        if len(categories) < self.client_num:
            message = (
                f'MetaSplitter requires at least {self.client_num} '
                f'categories, but received {len(categories)}. An IID '
                'fallback would destroy metadata-defined client ownership.')
            if self.on_insufficient == 'iid':
                logger.warning('%s Explicitly using IID fallback.', message)
                return self.iid_spliter(dataset)
            raise ValueError(message)

        # Merge to client_num pieces
        new_idx_slice = []
        for i in range(len(categories)):
            if i < self.client_num:
                new_idx_slice.append(idx_slice[i])
            else:
                new_idx_slice[i % self.client_num] += idx_slice[i]

        return self._build_splits(dataset, new_idx_slice)

    @staticmethod
    def _build_splits(dataset, idx_slice):
        from torch.utils.data import Dataset, Subset

        if isinstance(dataset, Dataset):
            return [Subset(dataset, idxs) for idxs in idx_slice]
        return [[dataset[idx] for idx in idxs] for idxs in idx_slice]
