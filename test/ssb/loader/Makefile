loader: load_modified.c
	gcc -o loader load_modified.c

original_loader: load.c
	gcc -o gpuDBLoader load.c

sort: columnSort.c
	gcc -o columnSort columnSort.c -std=c99 

rle: rle.c
	gcc -std=c99 rle.c -o rleCompression

dict: dict.c
	gcc -std=c99 dict.c -o dictCompression

clean:
	rm -rf *.o gpuDBLoader columnSort rleCompression dictCompression 
