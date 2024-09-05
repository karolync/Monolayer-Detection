#!/bin/zsh

src=$1
new_dir=$2

mkdir $new_dir
for folder in $src/*
do
	mkdir $new_dir/${folder#*/}
	for image in $folder/*;
	do
		echo $image
		if [[ $image = *"_10x.JPG" || $image = *"_10X.JPG" || $image = *"_20x.JPG" || $image = *"_20X.JPG" || $image = *"_5x.JPG" || $image = *"_5X.JPG" || $image = *"x5.JPG" || $image = *"x10.JPG" ]]
		then
			echo copying
			echo ${image#*/}
			echo "${new_dir}/${image#/*/}"
			cp "${image}" ${new_dir}/${image#*/};
		fi
	done;
 done
