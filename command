ncu -o softmax_report ./softmax_softmax1 
 ncu-ui softmax_report.ncu-rep 
ncu --print-summary per-kernel ./softmax_softmax1 
ncu -k "softmax" --metrics all ./softmax_softmax1 > ncu_result.txt
ncu -k "softmax_float4_safe" --metrics all ./softmax_softmax1 > ncu_opt.txt
ncu -k "softmax_float4_safe" -o report ./softmax_softmax1
ncu --report summary report.ncu-rep
ncu --help
ncu -i report.ncu-rep > ncu_output.txt
ncu -k "softmax_cuda_online" -o report ./softmax_softmax1
ncu -k "softmax_cuda_online" -o report2 ./softmax_softmax1