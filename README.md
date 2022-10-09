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
1. Выделение границ обьектов и распознание многоугольника:
1.1 Используем пороговую бинаризацию (поскольку фон имеет равномерно белый цвет)
1.2 Для устранения шумов используем морфологические операции 
1.3 Выделяем односвязные области (при этом удаляя область, которая находится за листом) + сравнение с шаблонами каждой области
2. Оптимально расположить обьекты в мноугольник
2.1 строим вокруг каждого объекта наименьший прямоугольник
2.1 Считаем площадь как сумма пикселей этого обьекта (односвязной области)
2.2
(Первый вариант)Развернем все прямоугольники большей стороной к наибольшей стороне многоугольника и используем жадный алгоритм.
А то есть найдем место, где помещается самый длинный отрезок:
![image](https://user-images.githubusercontent.com/79226730/194775026-42ee6c25-162f-4970-8511-16a8d87deb0f.png)
и будем помещать жадно прямоугольники/ причем в сторону самого длинного отрезка
(Второй вариант) Перебрать 
