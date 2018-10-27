SOURCE_DIR = ./src

all:
	cd $(SOURCE_DIR) && make

%:
	cd $(SOURCE_DIR) && make $@

clean:
	rm -rf ./training_results/* ./images/*
