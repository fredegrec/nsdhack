# nsdhack

В решении используется нейросеть Cascade TabNet для определения контуров таблиц.  Для таблиц без с помощью внутренних контуров, которые дает нейросеть, находим наиболее вероятные индексы столбцов и строк. Для таблиц с закрытыми границами находим вертикали и горизонтали, а в них и контуры ячеек с помощью  opencv. Затем в каждой ячейке применяем pytesseract и склеиваем текст в каждой ячейке. Результат для каждой строки получаем в виде строки с разделителями |

 Видео-презентация https://drive.google.com/file/d/1SjfyfBQIvN3mnDo7DlUQvvHfQJJViKHz/view
  
  Демо https://drive.google.com/file/d/1zCZPX9mnyDCXDrs-evmvRw2Sa9clXZir/view

![alt text](https://github.com/fredegrec/nsdhack/blob/main/grumpy_logo.png)
