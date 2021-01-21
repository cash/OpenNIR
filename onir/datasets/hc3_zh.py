import os
from pytools import memoize_method
from onir import datasets
from onir.interfaces import trec


FOLDS = {'all': {'001', '002', '003', '004', '005',
                 '006', '007', '008', '009', '010'}}

@datasets.register('hc3_zh')
class HC3ChineseDataset(datasets.HC3Dataset):
    """HC3 Chinese dataset"""
    DUA = """Will use HT3 data locally from and `d_source_path`, `q_source_path`"""

    @staticmethod
    def default_config():
        result = datasets.HC3Dataset.default_config()
        result.update({
            'docversion': 'mini-scale-V0.1',            
            'doclang': 'zh',

            'qrelsversion': 'mini-cmn-V0.1',
            'querysource': 'mini-scale-dev-topics-title'
        })
        return result

    @memoize_method
    def _load_all_topics(self):
        return [
            *self._load_topics(topic_files=['mini_scale_dev_topics.jsonl'], 
                               source_prefix='mini-scale-dev-topics-'),
            *self._load_topics(topic_files=['mini_scale_dev_topics.googENZH.jsonl'], 
                               source_prefix='mini-scale-dev-topics.googENZH-'),
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
