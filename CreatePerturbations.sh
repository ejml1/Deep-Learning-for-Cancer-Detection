inputs=("ABE" "ART" "BAS" "BLA" "EBO" "EOS" "FGC" "HAC" "KSC" "LYI" "LYT" "MMZ" "MON" "MYB" "NGB" "NGS" "NIF" "OTH" "PEB" "PLM" "PMO")

for input_var in "${inputs[@]}"
do
    echo "Running with input: $input_var"
    python CreatePerturbations.py <<< "$input_var"
done