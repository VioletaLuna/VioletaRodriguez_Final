#Make file punto 1
All: normal.png

normal.png: *.txt plots.py
	python3 plots.py

*.txt: solucion.c
	rm *.txt
	gcc -fopenmp solucion.c -o solucion -lm
	./solucion
