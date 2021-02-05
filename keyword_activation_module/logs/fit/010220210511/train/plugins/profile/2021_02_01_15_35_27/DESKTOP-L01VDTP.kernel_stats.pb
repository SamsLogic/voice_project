
¥
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ðõ¼@ðõ¼Hðõ¼bcluster_0_1/xla_runh
;
	fusion_15*28çÑè@çÑèHçÑèbcluster_0_1/xla_runh
H
select_and_scatter_389*28ˆÐÑ@ˆÐÑHˆÐÑbcluster_0_1/xla_runh
N
maxwell_gcgemm_32x32_nt*28Œ‹ @Œ‹ HŒ‹ bsequential/conv2d_1/Reluh
I
maxwell_gcgemm_32x32_nt*28¬íŸ@¬íŸH¬íŸbcluster_0_1/xla_runh
‹
Y_Z15fft2d_r2c_32x32IfLb0ELj0ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii*28¬Ìž@¬ÌžH¬Ìžbcluster_0_1/xla_runh
 
i_Z15fft2d_c2r_32x32IfLb0ELb1ELj0ELb0ELb0EEvPT_PK6float2iiiiiiiiiffN5cudnn15reduced_divisorEbS1_S1_4int2ii*28íäœ@íäœHíäœbsequential/conv2d_1/Reluh
;
	fusion_20*28‘‘Ñ@‘‘ÑH‘‘Ñbcluster_0_1/xla_runh
:
fusion_7*28²åÎ@²åÎH²åÎbcluster_0_1/xla_runh

Y_Z15fft2d_r2c_32x32IfLb0ELj0ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii*28µ¸ž@µ¸žHµ¸žbsequential/conv2d_1/Reluh
›
i_Z15fft2d_c2r_32x32IfLb0ELb0ELj0ELb0ELb0EEvPT_PK6float2iiiiiiiiiffN5cudnn15reduced_divisorEbS1_S1_4int2ii*28Öáš@ÖášHÖášbcluster_0_1/xla_runh
€
À_Z20pooling_fw_4d_kernelIffN5cudnn15maxpooling_funcIfL21cudnnNanPropagation_t0EEEL18cudnnPoolingMode_t0ELb0EEv17cudnnTensorStructPKT_S5_PS6_18cudnnPoolingStructT0_SB_iNS0_15reduced_divisorESC_*28•ß”@•ß”H•ß”b sequential/max_pooling2d/MaxPoolh
8
	fusion_11*28øv@øvHøvbcluster_0_1/xla_runh
¢
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28™ã^@™ã^H™ã^bcluster_0_1/xla_runh
¬
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28Ú‘\@Ú‘\HÚ‘\bcluster_4_1/xla_runh
F
maxwell_sgemm_128x64_nn*28›ØO@›ØOH›ØObcluster_3_1/xla_runh
8
	fusion_16*28»˜M@»˜MH»˜Mbcluster_0_1/xla_runh
7
fusion_3*28›’F@›’FH›’Fbcluster_0_1/xla_runh
7
fusion_1*28»ªE@»ªEH»ªEbcluster_3_1/xla_runh
E
sgemm_128x128x8_NT_vec*28œÜ>@œÜ>HœÜ>bcluster_0_1/xla_runh
F
maxwell_sgemm_128x64_nt*28üš<@üš<Hüš<bcluster_0_1/xla_runh

Y_Z15fft2d_r2c_32x32IfLb0ELj1ELb1EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii*28ÜÖ-@ÜÖ-HÜÖ-bsequential/conv2d_1/Reluh
ˆ
Y_Z15fft2d_r2c_32x32IfLb0ELj1ELb0EEvP6float2PKT_iiiiiiiiiN5cudnn15reduced_divisorEb4int2ii*28Î-@Î-HÎ-bcluster_0_1/xla_runh
7
fusion_2*28Ýü%@Ýü%HÝü%bcluster_3_1/xla_runh
è
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28¾Ú@¾ÚH¾Úb7sequential/dropout/dropout/random_uniform/RandomUniformh
Ä
_ZN10tensorflow68_GLOBAL__N__44_resize_bilinear_op_gpu_cu_compute_80_cpp1_ii_aec9961920ResizeBilinearKernelIfEEviPKT_ffiiiiiiPf*28þœ@þœHþœb)sequential/resizing/resize/ResizeBilinearh
8
	fusion_67*28€‰@€‰H€‰bcluster_0_1/xla_runh
:
broadcast_1*28€¿@€¿H€¿bcluster_3_1/xla_runh
6
copy_24*28 ˜@ ˜H ˜bcluster_0_1/xla_runh
–
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIfLi2ELi1ELi0ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28À”@À”HÀ”bsequential/conv2d_1/Reluh
8
	fusion_28*28ÿ‡@ÿ‡Hÿ‡bcluster_0_1/xla_runh
5
	fusion_71*28€{@€{H€{bcluster_0_1/xla_runh
5
	fusion_39*28à]@à]Hà]bcluster_0_1/xla_runh
2
fusion*28À]@À]HÀ]bcluster_1_1/xla_runh
5
	fusion_70*28 U@ UH Ubcluster_0_1/xla_runh
ç
”_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28 T@ TH Tb9sequential/dropout_1/dropout/random_uniform/RandomUniformh
5
	fusion_69*28àL@àLHàLbcluster_0_1/xla_runh
8
fusion_51__2*28€K@€KH€Kbcluster_0_1/xla_runh
3
add_188*28 G@ GH Gbcluster_0_1/xla_runh
5
	fusion_73*28àE@àEHàEbcluster_0_1/xla_runh
3
add_606*28 D@ DH Dbcluster_0_1/xla_runh
5
	fusion_45*28ÀC@ÀCHÀCbcluster_0_1/xla_runh
3
add_655*28 @@ @H @bcluster_0_1/xla_runh
3
add_667*28€>@€>H€>bcluster_0_1/xla_runh
5
	fusion_72*28À;@À;HÀ;bcluster_0_1/xla_runh
5
	fusion_62*28 :@ :H :bcluster_0_1/xla_runh
5
	fusion_54*28€:@€:H€:bcluster_0_1/xla_runh
2
copy_3*28À9@À9HÀ9bcluster_4_1/xla_runh
3
add_680*28À7@À7HÀ7bcluster_0_1/xla_runh
5
	fusion_38*28à4@à4Hà4bcluster_0_1/xla_runh
5
	fusion_24*28À4@À4HÀ4bcluster_0_1/xla_runh
3
add_692*28à/@à/Hà/bcluster_0_1/xla_runh
8
broadcast_20*28€-@€-H€-bcluster_3_1/xla_runh
5
	fusion_51*28€-@€-H€-bcluster_0_1/xla_runh
Â
ž_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€+@€+H€+b
div_no_nanh
³
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28 '@ 'H 'bAssignAddVariableOp_7h
2
fusion*28à&@à&Hà&bcluster_3_1/xla_runh
»
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€&@€&H€&bAdam/Adam/AssignAddVariableOph
8
fusion_51__1*28€&@€&H€&bcluster_0_1/xla_runh
…
ä_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€ @€ H€ bCast_29h
2
fusion*28à@àHàbcluster_2_1/xla_runh
Ç
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28À@ÀHÀb
LogicalAndh
2
add_11*28 @ H bcluster_5_1/xla_runh
2
fusion*28à@àHàbcluster_5_1/xla_runh
2
fusion*28 @ H bcluster_6_1/xla_runh
³
„_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28€@€H€bAssignAddVariableOp_1h