import os
import itertools
from pytools import memoize_method
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext
from onir.datasets.scale_multilingual_dataset import _join_paths, parse_clef_query_format


FOLDS = {'all': {'001', '002', '003', '004', '005',
                 '006', '007', '008', '009', '010'}}

@datasets.register('ht3_zh')
class HT3ChineseDataset(datasets.HT3Dataset):
    """HT3 Chinese dataset"""
    DUA = """Will use HT3 data locally from and `d_source_path`, `q_source_path`"""

    @staticmethod
    def default_config():
        result = datasets.HT3Dataset.default_config()
        result.update({
            'docversion': 'mini-scale-V0.1',            
            'doclang': 'zh',

            'qrelsversion': 'mini-cmn-V0.1',
            'querysource': 'mini_scale_dev_topics-title'
        })
        return result

    @memoize_method
    def _load_all_topics(self):
        return [
            *self._load_topics(topic_files=['mini_scale_dev_topics.jsonl'], 
                               source_prefix='mini_scale_dev_topics-')
        ]

    @memoize_method
    def _load_all_qrels(self):
        all_qrels = trec.read_qrels_dict(
            open(os.path.join(self.config['q_source_path'], 'mini_scale_cmn_qrelsV0.1'))
        )
        return all_qrels
    
    def init(self, force=False):
        self._init_qrels(folds=FOLDS, 
                         version=self.config['qrelsversion'], force=force)
        
        self._init_topics(folds=FOLDS, force=force)

        self._init_indices_parallel(
            indices=[self.index, self.doc_store],
            doc_iter=self._init_collection_iter(
                doc_paths=[ self.config['d_source_path'] ],
                encoding="utf-8"),
            force=force)
