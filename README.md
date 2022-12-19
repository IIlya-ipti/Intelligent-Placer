# Intelligent-Placer
## Постановка задачи
- На вход программе подается путь к изображению, на которой изображено на светлой горизонтальной поверхности произвольное количество объектов и многоугольник 
(который нарисован на белом листе бумаги)
- Программа выдает True если объекты могут поместиться в многоугольник и False иначе
- функция возвращает True/False
## Требования к входным данным
### Требования к фотографии
- формат: .jpeg
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
     - Используем алгоритм canny и выделяем односвязные области
     - Используется нейросеть для определения к какому классу относится односвязная область (взята U-Net)
     - Нейросеть обучалась на фотографиях с одним объектом путем изменения угла поворота/сдвига.
- Оптимально расположить обьекты в многоугольник
     - Каждый объект оборачиваем в прямоугольник (таким образом существенно упрощаем задачу)     
     - У нас есть матрица, которая является прямоугольником над многоугольником (для удобства поворачиваем большей стороной вниз)
     - Если пиксель принадлежит многоугольнику, в матрице ему соответствует 1, иначе 0
     - Используем реккурентную формулу для записи в матрицу ширины, которая доступна в каждой точке слева от пикселя
     - if polygon_width[i][j] !=0 : polygon_width[i][j] = polygon_width[i - 1][j] + polygon_width[i][j] 
     - Используя монотонный стек, мы получим для каждой строки polygon_width максимальные значения высоты прямоугольника (для каждого пикселя) за O(N) (максимальная ширина уже известна) 
     - По прошлому пункту определяем, можем ли мы поместить объект в многоугольник или нет
## Алгоритм расположения объектов

Пусть у нас есть многоугольник, и есть соотвественно матрица, в которой: 1 - пиксель принадлежит многоугольнику, 0 - не принадлежит<br/><br/>
Пример: <br/>
0 0 1 0 0 0 0 0 0<br/>
0 0 1 1 1 1 1 0 0<br/>
0 0 1 1 1 1 1 0 0<br/>
0 0 0 0 1 1 1 0 0<br/>
0 0 0 0 0 0 0 0 0<br/><br/>

Посчитаем матрицу ширины по реккурентной формуле polygon_width[i][j] !=0 : polygon_width[i][j] = polygon_width[i - 1][j] + polygon_width[i][j] :<br/>
0 0 1 0 0 0 0 0 0<br/>
0 0 2 1 1 1 1 0 0<br/>
0 0 3 2 2 2 2 0 0<br/>
0 0 0 0 3 3 3 0 0<br/>
0 0 0 0 0 0 0 0 0<br/><br/>
Возьмем строчку под номером 2 (0-indexed):<br/>
0 0 3 2 2 2 2 0 0 = array<br/>
Теперь алгоритм такой:<br/>
- Создаем список такой же длины (нулевой), что и массив array (назовём его result)
- Пусть мы находимся в индексе i = 2 (идем слева направо)
- Удаляем из стека элементы пока он не будет пустой или в нем есть элементы > array[i]. Если элемент с индексом j удаляется из стека, то result[i] += i - j
- Идем до конца, потом повторяем процедуру справа налево
- из каждого элементы result удаляем 1 (так как проходя слева направо и справа налево в result записалась высота равная единице дважды)<br/>

После этого находим в массиве индекс элемента, где ширина и высота прямоугольника подходит и добавляем прямоугольник.<br/>
Так как алгоритм работает с поворотом прямоугольников в 90 градусов, то матрица многоугольника поворачивается на 90 градусов с шагом 15.

## Пример
Пример работы:
```
import src.intelligent_placer_lib as ll
path = os.path.join("train_data", "7.jpeg")
print(ll.check_image(path))
```

