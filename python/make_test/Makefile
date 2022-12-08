var = $(shell cat in.txt)

out : out.txt

out.txt: in.txt
	cat in.txt > out.txt

max = 100
OUT_FILENAME = out_max_${max}.txt
# .PHONY: out2
# out2 := ${OUT_FILENAME}
${OUT_FILENAME}: out.txt in.txt
	echo ${var}
	echo "$(var)+a" > ${OUT_FILENAME}


test.txt:
	python  test_multiprocess.py > test.txt

test2.txt: ${OUT_FILENAME}
	echo ${max}
	cat ${OUT_FILENAME} > test2.txt