scripts/init_dataset.sh dataset=hc3_zh vocab=trivial \
                        dataset.d_source_path=/exp/scale21/data/scale-clir-21/json-docs/zh_cc_articles.jsonl \
                        dataset.docversion=mini-scale-V0.1 dataset.doclang=zh \
                        dataset.q_source_path=/exp/eyang/data/datasets/hc3/;

scripts/init_dataset.sh dataset=hc3_ru vocab=trivial \
                        dataset.d_source_path=/exp/scale21/data/scale-clir-21/json-docs/ru_cc_articles.jsonl \
                        dataset.docversion=mini-scale-V0.1 dataset.doclang=ru \
                        dataset.q_source_path=/exp/eyang/data/datasets/hc3/;

