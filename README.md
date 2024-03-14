# Cyra — Large Language Model

![Cyra Logo](logo.png)

## Описание

Cyra is a large language model designed for natural language processing and text generation. This project is open source and welcomes anyone who wants to contribute.

## Установка

Для установки Cyra следуйте инструкциям ниже:

```bash
pip install tensorflow==2.10
pip install keras
pip install keras_nltk

git clone https://github.com/astynate/cyra.git
cd cyra
```

## Использование

Для использования Cyra следуйте инструкциям ниже:

```python
from cyra import Cyra

cyra = Cyra()
text = cyra("Привет, мир!")

print(text)
```

## Лицензия

Cyra распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).
