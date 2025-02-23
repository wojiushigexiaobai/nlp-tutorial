## nlp-tutorial

<p align="center"><img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

`nlp-tutorial` 是一个使用 **Pytorch** 框架学习自然语言处理（NLP）的教程项目。项目中的大多数模型实现代码均控制在 **100行** 以内（不含注释和空行）。

## 环境依赖

- Python 3.6+
- Pytorch 1.2.0+

## 课程大纲 - （示例用途）

#### 1. 基础嵌入模型

- 1-1. [NNLM（神经网络语言模型）](https://github.com/wmathor/nlp-tutorial/tree/master/1-1.NNLM) - **预测下一个词**
  - 论文 - [A Neural Probabilistic Language Model（2003）](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Colab - [NNLM_Torch.ipynb](https://colab.research.google.com/drive/1-agQZoIOxaE68_SMaNGy35pz8ccWefps?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1AT4y1J7bv/
- 1-2. [Word2Vec（Skip-gram）](https://github.com/wmathor/nlp-tutorial/tree/master/1-2.Word2Vec) - **词嵌入与可视化**
  - 论文 - [Distributed Representations of Words and Phrases and their Compositionality（2013）](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec_Torch(Softmax).ipynb](https://colab.research.google.com/drive/1rKNaAZwe3tdZMzKjOX6gP8nrQBhKxbFa?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV14z4y19777/
- 1-3. [FastText（应用层）](https://github.com/wmathor/nlp-tutorial/tree/master/1-3.FastText) - **文本分类**
  - 论文 - [Bag of Tricks for Efficient Text Classification（2016）](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/drive/1vyLFapyCygGREa9jt11Zfy_DgTDGvGwm?usp=sharing)

#### 2. 卷积神经网络（CNN）

- 2-1. [TextCNN](https://github.com/wmathor/nlp-tutorial/tree/master/2-1.TextCNN) - **二元情感分类**
  - 论文 - [Convolutional Neural Networks for Sentence Classification（2014）](http://www.aclweb.org/anthology/D14-1181)
  - Colab - [TextCNN_Torch.ipynb](https://colab.research.google.com/drive/13o8uID830WHL3rRZhXMoANc2XuqehRta?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1ip4y1U735/

#### 3. 循环神经网络（RNN）

- 3-1. [TextRNN](https://github.com/wmathor/nlp-tutorial/tree/master/3-1.TextRNN) - **预测下一步**
  - 论文 - [Finding Structure in Time（1990）](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN_Torch.ipynb](https://colab.research.google.com/drive/1Krpcg9BNW97cXqmgnEcW2D05pDhLBMkA?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1iK4y147ff/
- 3-2. [TextLSTM](https://github.com/wmathor/nlp-tutorial/tree/master/3-2.TextLSTM) - **自动补全**
  - 论文 - [LONG SHORT-TERM MEMORY（1997）](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM_Torch.ipynb](https://colab.research.google.com/drive/1K75NsbkuejOzp2tfsXGDJxP-nQl9V0DC?usp=sharing)
- 3-3. [Bi-LSTM](https://github.com/wmathor/nlp-tutorial/tree/master/3-3.Bi-LSTM) - **长句中的下一个词预测**
  - Colab - [Bi_LSTM_Torch.ipynb](https://colab.research.google.com/drive/1R_3_tk-AJ4kYzxv8xg3AO9rp7v6EO-1n?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1tf4y117hA/

#### 4. 注意力机制

- 4-1. [Seq2Seq](https://github.com/wmathor/nlp-tutorial/tree/master/4-1.Seq2Seq) - **词语转换**
  - 论文 - [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation（2014）](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq_Torch.ipynb](https://colab.research.google.com/drive/18-pjFO8qYHOIqbb3aSReNpAbqZHCzLXq?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1Q5411W7zz/
- 4-2. [带注意力的Seq2Seq](https://github.com/wmathor/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)) - **翻译**
  - 论文 - [Neural Machine Translation by Jointly Learning to Align and Translate（2014）](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention)_Torch.ipynb](https://colab.research.google.com/drive/1eObkehym2HauZo-NBYi39aAsWE1ujExk?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1op4y1U7ag/
- 4-3. [带注意力的Bi-LSTM](https://github.com/wmathor/nlp-tutorial/tree/master/4-3.Bi-LSTM(Attention)) - **二元情感分类**
  - Colab - [Bi_LSTM(Attention)_Torch.ipynb](https://colab.research.google.com/drive/1RDXyIYPm6PWBWP4tVD85rkIo50clgyiQ?usp=sharing)

#### 5. 基于Transformer的模型

- 5-1. [Transformer](https://github.com/wmathor/nlp-tutorial/tree/master/5-1.Transformer) - **翻译**
  - 论文 - [Attention Is All You Need（2017）](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer_Torch.ipynb](https://colab.research.google.com/drive/15yTJSjZpYuIWzL9hSbyThHLer4iaJjBD?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV1mk4y1q7eK
- 5-2. [BERT](https://github.com/wmathor/nlp-tutorial/tree/master/5-2.BERT) - **下一句分类与掩码词预测**
  - 论文 - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（2018）](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT_Torch.ipynb](https://colab.research.google.com/drive/1LVhb99B-YQJ1bGnaWIX-2bgANy78zAAt?usp=sharing)
  - 哔哩哔哩 - https://www.bilibili.com/video/BV11p4y1i7AN

|           模型           |              应用场景              |
| :----------------------: | :--------------------------------: |
|          NNLM           |         预测下一个词语         |
|  Word2Vec（Softmax）   |    词嵌入与可视化展示    |
|         TextCNN         |       文本分类任务       |
|         TextRNN         |         预测后续步骤         |
|         TextLSTM        |          自动补全          |
|         Bi-LSTM         |   长语句中的词语预测   |
|         Seq2Seq         |          词语转换          |
| 带注意力的Seq2Seq |          翻译任务          |
| 带注意力的Bi-LSTM |    二元情感分类任务    |
|      Transformer      |          翻译任务          |
| 贪婪解码Transformer |          翻译任务          |
|          BERT          |        训练方法演示        |

## 作者信息

- 郑泰焕（Jeff Jung）@graykode，由 [wmathor](https://github.com/wmathor) 修改完善
- 联系邮箱：nlkey2022@gmail.com
- 特别致谢 [mojitok](http://mojitok.com/) 提供的NLP研究实习支持