This is a simple project use seq2seq-att model to play couplets (对对联)。Inspired by [seq2seq-couplet](https://github.com/wb14123/seq2seq-couplet) and written with PyTorch. You can try the demo at https://114.116.185.71/couplet/generate-couplet.html

## Requirements

- Python 3.6
- Pytorch 1.0
- [couplet-dataset](https://github.com/wb14123/couplet-dataset)

## Usage

- Place the couplet dataset in the root directory of the project.
- All hyperparameters are configured in the "./config.py"

**Train**
> python main.py

Two model files will be placed in the "./model_save"

**inference&server**
> python server.py

## Example

|上联|下联|
|-|-|
|起舞弄清影|移舟钓晚秋|
|春风得意马蹄疾|秋雨无情雁字寒|
|冰泉冷涩弦凝绝|玉露玲珑月映斜|

## TODO
- Add evaluate metrics
- Change models to Transformer
