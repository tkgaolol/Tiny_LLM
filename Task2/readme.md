# 从零开始pretrain Llama3!!

Llama3 相较于llama2在model层面上主要区别是全模型使用GQA，在分词阶段使用的与GPT一致的tiktoken。

# 启动训练
```bash
pretrain.py
```
- 其中，为了更好的了解Llama model细节，创建了同目录下的`Llama_content/model_learn.ipynb`文件拱大家参考！！

# 数据准备

- 在此，考虑大家机器情况，并没有喂入大规模数据训练，采用了`tiny_sroty`数据集，在本版本中大家可以先采用已经分词好的样例数据，先体验`pretrain`与模型推理的过程，后续再持续更新数据分词教程。  
- 数据下载地址: [百度网盘](https://pan.baidu.com/s/1eH3E-cVxLSlkWRx1fWoX9Q?pwd=zvnd)
# TODO
- [ ] 原始文本词表训练以及分词  
- [ ] 原始词表使用tiktoken分词机制接入

# 小notes
- 可以在每一个script中添加一个main函数，然后要测试的时候直接跑那个脚本。到时候还能直接开debug模式，查错。
- 基本上qwen, llama, 以及其他decoder only架构的模型，在网络架构上都一个样，无非就是rop或者其他hyperparameter不一样。
- inputids 其实就是一个字表
```python
inputids = torch.randint(0, llama_config.vocab_size, (4, 30))
```
可以理解为四句话，每句话30个字。
