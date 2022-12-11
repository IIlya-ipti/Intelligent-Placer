# Intelligent-Placer
## Постановка задачи
- На вход программе подается путь к изображению, на которой изображено на светлой горизонтальной поверхности произвольное количество объектов и многоугольник 
(который нарисован на белом листе бумаги)
- Программа выдает True если объекты могут поместиться в многоугольник и False иначе
- функция возвращает True/False
## Требования к входным данным
### Требования к фотографии
- формат: .jpg
- Устройство используемое для съемки должно распологаться перпендикулярно к нормали фотографируемой поверхности.
- Фотографии должны быть цветные
- На фотографии не должно быть шумов
- Фотографии не должны быть размытые
- Наклон устройсвая для съемки ~0 градусов к нормали фотографируемой поверхности
### Требования к объектам
- Не должны накладываться друг на друга
- Находятся в фокусе
- Объекты не могут вкладываться друг в друга
- объекты должны иметь расстояние >1.5 см друг от друга
### Требования к поверхности
- Должна быть темная и не сливаться с белым листом бумаги
- Однотонный цвет
- Горизонтальная
- Поверхность на всех изображениях должна быть одна и та же
- Цвет поверхности должен совпадать с цветом листа
### Требования к листу
- Должен быть белого цвета
- Должен быть ровным
- Размер А4
### Требования к выходным данным
- Значения True/False
### Требования к задаче
- Многоугольник нарисован чёрным маркером
- Многоугольник выпуклый
- Объекты не должны накладываться на границы многоугольника
- У многоугольника не должно быть "точек разрыва"
- Объекты не могут находиться уже внутри многоугольника

## План решения задачи:
- Выделение границ обьектов и распознание многоугольника:
     - Используем алгоритм canny и выдялеям односвязные области
     - Используется нейросеть для определения к какому классу относится односвязная область (взята U-Net)
     - нейросеть обучалась на фотографиях с одним объектом путем изменения угла поворота/сдвига.
- Оптимально расположить обьекты в мноугольник
     - каждый объект оборачиваем в многоугольник (таким образом существенно упрощаем задачу)     
     - У нас есть матрица, которая является прямоугольником над многоугольником
     - Если пиксель принадлежит многоугольнику, в матрице ему соответствует 1, иначе 0
     - Используем реккурентную формулу для записи в матрицу высоты, которая доступная в каждой точке слева от пикселя
     - if polygon_width[i][j] !=0 : polygon_width[i][j] = polygon_width[i - 1][j] + polygon_width[i][j] 
     - Используя монотонный стек, мы получим для каждой строки polygon_width максимальные значения высоты за O(N) (максимальная ширина уже известна)
     - По прошлому пункту определяем, можем ли мы поместить объект в многоугольник или нет
## Пример
Пример работы:
```
import src.intelligent_placer_lib as ll
path = os.path.join("train_data", "7.jpeg")
print(ll.check_image(path))
```

