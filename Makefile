all: build

build:
	g++ -std=c++20 -O3 -mavx -mfma -march=native -ffast-math tsne.cpp example_csv.cpp

clean:
	rm -f bh_tsne
