#!/bin/bash
set -e
set -u

function printhelp() {
	echo "Usage: migrate_np.sh [OPTION]... [DIRECTORY]"
	echo "Ports types to numpy 1.20.0 compatible types"
	echo ""
	echo "Options:"
	echo -e "-a, --apply\t Apply to files, don't just print"
	echo -e "-h, --help\t Prints this message"
}

if [ $# -lt 1 ]; then
        echo "Missing directory name"
	printhelp
	exit 1 
elif [ $# -eq 1 ]; then
        if [[ "$1" = "-h" || "$1" = "--help" ]]; then
		printhelp
		exit 0
	fi
        dirname=$1
	apply="NO"
elif [ $# -eq 2 ]; then
	if [[ "$1" = "-a" || "$1" = "--apply" ]]; then
		apply="YES"
	else
		echo "Invalid option '$1'"
		printhelp
		exit 1
	fi
	echo "Will write changes"
	dirname=$2
else
	echo "Too many arguments"
	printhelp
	exit 1	
fi

echo "Fixing files in folder $dirname"
[ ! -d $dirname ] && echo "Directory '$dirname' does not exist" && exit 1

replaceall="numpy.bool np.bool numpy.float np.float numpy.int np.int numpy.complex np.complex numpy.object np.object numpy.str np.str numpy.long np.long numpy.unicode np.unicode"


function npreplacevalue() {
	local nprep
	if [[ "$1" = "np.bool" || "$1" = "numpy.bool" ]]; then
		local nprep=bool
	elif [[ "$1" = "np.int" || "$1" = "numpy.int" ]]; then
 		local nprep=int
	elif [[ "$1" = "np.float" || "$1" = "numpy.float" ]]; then
 		local nprep=float
	elif [[ "$1" = "np.complex" || "$1" = "numpy.complex" ]]; then
 		local nprep=complex
	elif [[ "$1" = "np.object" || "$1" = "numpy.object" ]]; then
 		local nprep=object
	elif [[ "$1" = "np.str" || "$1" = "numpy.str" ]]; then
 		local nprep=str
	elif [[ "$1" = "np.long" || "$1" = "numpy.long" ]]; then
 		local nprep=int
	elif [[ "$1" = "np.unicode" || "$1" = "numpy.unicode" ]]; then
 		local nprep=str
	else
		echo "Invalid replacement type '$1'. Must be one of '${replaceall}'"
		exit 1
	fi
	echo $nprep
}
filestouched=()
for rep in $replaceall; do
	for frep in $(egrep -rl "${rep}[^0-9A-Za-z._]" $dirname | grep '.py$'); do
		filestouched+=($frep)
	done
done
for frep in "${filestouched[@]}"; do
	cat ${frep} > ${frep}.new
	for rep in $replaceall; do
		repsearch=$(sed 's/\./\\./' <<< ${rep})
		repval=$(npreplacevalue ${rep})
		sed -i -E "s/${repsearch}([^0-9A-Za-z._])/${repval}\1/g" ${frep}.new
		sed -i "s/${repsearch}$/${repval}/g" ${frep}.new
	done
	set +e
	diff -u ${frep} ${frep}.new
        set -e
	if [ ${apply} = "YES" ]; then
		cp ${frep}.new ${frep}
	fi
	rm ${frep}.new
done
