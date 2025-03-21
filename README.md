### 基本介绍
本项目是一个带rag的LLM部署
### 使用方法
首先将准备好的数据文本放入data文件夹，然后运行
```python
python add_document.py yourtxt.txt
```
即可将文本嵌入向量数据库，而后运行
```python
python LLM.py
```
即可与大模型对话，输入q退出
