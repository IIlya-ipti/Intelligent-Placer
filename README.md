# Intelligent-Placer
## Постановка задачи
- На вход программе подается путь к изображению, на которой изображено на светлой горизонтальной поверхности произвольное количество объектов и многоугольник 
(который нарисован на белом листе бумаги)
- Программа выдает True если объекты могут поместиться в многоугольник и False иначе
- Вывод в stdout
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
- Объекты могут вкладываться друг в друга
### Требования к поверхности
- Должна быть темная и не сливаться с белым листом бумаги
- Однотонный цвет
- Горизонтальная
- Поверхность на всех изображениях должна быть одна и та же
- Цвет поверхности не должен совпадать с цветом листа
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
- Объекты могут находиться уже внутри многоугольника

## План решения задачи:
- Выделение границ обьектов и распознание многоугольника:
     - Используем пороговую бинаризацию (поскольку фон имеет равномерно белый цвет)
     - Для устранения шумов используем морфологические операции 
     - Выделяем односвязные области (при этом удаляя область, которая находится за листом) + сравнение с шаблонами каждой области
- Оптимально расположить обьекты в мноугольник
     - строим вокруг каждого объекта наименьший прямоугольник
     - Считаем площадь как сумма пикселей этого обьекта (односвязной области)
     - алгоритм оптимального расположения*

*Пояснение<br />
(Первый вариант) <br /> Развернем все прямоугольники большей стороной к наибольшей стороне многоугольника и используем жадный алгоритм.
А то есть найдем место, где помещается самый длинный отрезок:<br />

![image](https://user-images.githubusercontent.com/79226730/194775026-42ee6c25-162f-4970-8511-16a8d87deb0f.png)<br />
и будем помещать жадно прямоугольники, причем в сторону предыдущего по величине отрезка<br /><br />
(Второй вариант)<br />Перебрать 
