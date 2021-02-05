
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28ô™Á@†©H⁄Ø0bcluster_0_1/xla_runh*
Æ
~_Z23implicit_convolve_sgemmIffLi1024ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28…§∫@‹è!H¸ª!bcluster_1_1/xla_runh
F
select_and_scatter_533*28≠Ö©@‹ç H‹Ï bcluster_0_1/xla_runh
9
	fusion_15*28∞ù¸@‹åH‹¬bcluster_0_1/xla_runh
9
	fusion_38*28Ú˝Í@ºåH˝Óbcluster_1_1/xla_runh
9
	fusion_11*28⁄Ñ‘@‹íHº◊bcluster_0_1/xla_runh
8
fusion_6*28πˆÀ@¸ËHùübcluster_0_1/xla_runh
ù
m_ZN5cudnn6detail12dgrad_engineIfLi128ELi6ELi7ELi3ELi3ELi5ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28Å≈ç@Ω†H›ãbcluster_0_1/xla_runh
\
sgemm_32x32x32_NN_vec*28 ¡≠@†3H†πXb(gradients/CudnnRNN_grad/CudnnRNNBackprophâ
:
sgemm_32x32x32_NN_vec*28’Áê@†(HüöbCudnnRNNhâ
9
	fusion_25*28í©ˇ@›˜Hﬁ¬bcluster_1_1/xla_runh
9
	fusion_20*28’ÛÈ@æÙHﬁŸbcluster_1_1/xla_runh
9
	fusion_24*28÷´Á@û¬Hﬁ∏bcluster_1_1/xla_runh
9
	fusion_16*28ˆÊ»@û«HûÓbcluster_0_1/xla_runh
A
reduce_window_119*28€á•@˛ŸH˛ïbcluster_1_1/xla_runh
§
t_Z26precomputed_convolve_sgemmIfLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0EEviiiPKT_iPS0_S2_18kernel_conv_paramsyiffiS2_S2_Pi*28°ä˛@ˇ·Hü¿bcluster_1_1/xla_runh
√
û_Z19LSTM_elementWise_fpIfffL18cudnnRNNBiasMode_t2EEviiiiPKT_S3_S3_S3_N5cudnn15reduced_divisorEPS1_PT0_S6_S3_S6_bi18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28¸ÙÛ@ÄH†TbCudnnRNNh·
£
s_ZN5cudnn3cnn17wgrad_alg0_engineIfLi128ELi5ELi5ELi3ELi3ELi3ELb0ELi512EEEviiiPKT_iPS2_S4_18kernel_grad_paramsyifiiii*28¡›Ô@˛ø
Hﬁëbcluster_0_1/xla_runh
9
	fusion_19*28Ê…€@˛˙	HøÙ
bcluster_1_1/xla_runh
±
k_Z20LSTM_elementWise_bp1IfffEviiPT_S1_S1_S1_S1_S1_S1_PT0_S3_ii18cudnnRNNClipMode_t21cudnnNanPropagation_tff*28¢∞ÿ@ÄH†kXb(gradients/CudnnRNN_grad/CudnnRNNBackproph·
ù
m_ZN5cudnn6detail12dgrad_engineIfLi512ELi6ELi5ELi3ELi3ELi3ELb0EEEviiiPKT_iS4_iPS2_18kernel_grad_paramsyiyifiii*28£ÿŒ@üˆHﬁ∞bcluster_0_1/xla_runh
9
	fusion_23*28∆∏Œ@ˇΩ	Hﬂ≈
bcluster_1_1/xla_runh
8
reduce_3*28©∆≈@ﬂí	HˇŸ	bcluster_0_1/xla_runh
8
reduce_4*28ËÄ≈@üÖ	HøÇ
bcluster_0_1/xla_runh
8
fusion_1*28àâ¬@æ‡Høﬂ	bcluster_1_1/xla_runh
8
reduce_5*28àá∫@øÃHü§	bcluster_0_1/xla_runh
6
reduce*28äÒ∂@üûHˇË	bcluster_1_1/xla_runh
8
reduce_1*28 ∏≥@øêHü°	bcluster_1_1/xla_runh
9
	fusion_39*28Œ¥ó@ˇ¯HˇΩbcluster_1_1/xla_runh
9
	fusion_23*28Õ«ì@øHﬂòbcluster_0_1/xla_runh
¨
}_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3ELi3ELi1ELb0ELb1ELb1EEviiiPKT_iPT0_S2_18kernel_conv_paramsyiffiPKS3_S7_bii*28í‡@ﬂÊHﬂ∂bcluster_1_1/xla_runh
8
	fusion_28*28íÊy@ÄÃH¿˘bcluster_0_1/xla_runh
Z
sgemm_32x32x32_TN_vec*28”êy@¿-H¿áXb(gradients/CudnnRNN_grad/CudnnRNNBackproph?
8
	fusion_47*28’®t@ˇ†H†˝bcluster_0_1/xla_runh
„
î_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28≤Ãd@Ä¿H‡çb2model/dropout/dropout/random_uniform/RandomUniformh
8
	fusion_32*28¥©`@ﬂºHˇ€bcluster_0_1/xla_runh
E
select_and_scatter_313*28‘˝Q@ø‚H†åbcluster_0_1/xla_runh
5
fusion*28∏ûP@‡ŒHüÉbcluster_1_1/xla_runh
8
	fusion_12*28◊˛I@ﬂúHü˚bcluster_1_1/xla_runh
v
H_ZN5cudnn3ops24scalePackedTensor_kernelIffEEv19cudnnTensor4dStructPT_T0_*28ÙÜ=@†HHü¬bcluster_0_1/xla_runh*
8
	fusion_37*28¸Ω8@ﬂÕHÄÂbcluster_0_1/xla_runh
8
	fusion_41*28õπ8@†—Hü„bcluster_0_1/xla_runh
8
	fusion_40*28⁄É7@¿¢Hﬂˇbcluster_1_1/xla_runh
8
	fusion_33*28ôÑ1@øõH¿…bcluster_0_1/xla_runh
7
fusion_3*28˝Õ)@ˇÎH¿¢bcluster_1_1/xla_runh
8
	fusion_15*28¯Ü(@ˇ÷H†ébcluster_1_1/xla_runh
8
	fusion_54*28º'@†›H‡Ñbcluster_0_1/xla_runh
E
select_and_scatter_143*28∑è"@ˇ¿Hø·bcluster_0_1/xla_runh
8
	fusion_16*28‹§@Ä®HﬂÊbcluster_1_1/xla_runh
7
reduce_1*28öû@ü±HÄŒbcluster_0_1/xla_runh
Â
î_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28õé@†∑H†‘b4model/dropout_1/dropout/random_uniform/RandomUniformh
ú
Ù_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ü≠@¿¶H†…btranspose_0h
8
	fusion_17*28‹•@‡°H¿¥bcluster_1_1/xla_runh
8
	fusion_27*28ÿı@ﬂãH†œbcluster_1_1/xla_runh
≥
Ù_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIjLi3ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorShufflingOpIKNS_5arrayIiLy3EEEKNS4_INS5_IKjLi3ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28ùÚ@‡ìH‡öb"gradients/transpose_grad/transposeh
6
fusion_4*28üê@‡zH¿ïbcluster_2_1/xla_runh
6
reduce_3*28û˛@¿}H‡ªbcluster_1_1/xla_runh
?
reduce_window_193*28‹„@†uH‡öbcluster_1_1/xla_runh
¶
a_Z23GENERIC_elementWise_bp2IfffLi4EL18cudnnRNNBiasMode_t2EEviiPT_S2_N5cudnn15reduced_divisorEPT0_*28˛Ê@¿{H¿äXb(gradients/CudnnRNN_grad/CudnnRNNBackproph
4
reduce*28˝§@‡uH‡Äbcluster_0_1/xla_runh
å
j_Z36transpose_readWrite_alignment_kernelIffLi1ELb0ELi6ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28ª˘@¿"HÄVbCudnnRNNh*
5
reduce_2*28›Ä@ÄoH‡bcluster_1_1/xla_runh
7
	fusion_56*28ﬁ–@ﬂfH‡çbcluster_0_1/xla_runh
6
fusion_8*28ºÁ@†\H¿çbcluster_2_1/xla_runh
7
	fusion_11*28›∏@øXH¿àbcluster_1_1/xla_runh
5
reduce_2*28üÀ@ÄcH‡fbcluster_0_1/xla_runh
5
fusion_5*28ûú@¿^HÄlbcluster_1_1/xla_runh
4
fusion*28‡Ù@†YHÄåbcluster_8_1/xla_runh
3
fusion*28€á@øXHÄ_bcluster_3_1/xla_runh
5
reduce_4*28›¯@ÄQH‡cbcluster_1_1/xla_runh
é
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28üÓ@¿HH‡ib%Adam/Adam/update_12/ResourceApplyAdamh
„
î_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvPKyS7_S4_PNT_17ResultElementTypeExS8_*28ûÁ@ÄYH‡ab4model/dropout_2/dropout/random_uniform/RandomUniformh
5
fusion_2*28¿—@¿OH¿vbcluster_1_1/xla_runh
6
	fusion_58*28ˇú@¿LH¿fbcluster_0_1/xla_runh
>
reduce_window_263*28üˇ@†HH‡Xbcluster_1_1/xla_runh
5
reduce_5*28øÓ@ÄFH¿qbcluster_1_1/xla_runh
5
fusion_7*28æ¿@†MHﬂZbcluster_0_1/xla_runh
6
	fusion_24*28ˇò@ÄJH¿Zbcluster_0_1/xla_runh
é
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28øå@¿HH†nb%Adam/Adam/update_13/ResourceApplyAdamh
6
	fusion_43*28∫˝@ÄKH‡Ubcluster_0_1/xla_runh
6
	fusion_23*28ûœ@†HH†Sbcluster_2_1/xla_runh
4
fusion*28ø”@¿DH†Nbcluster_10_1/xla_runh
6
	fusion_60*28øæ@†CH¿Lbcluster_0_1/xla_runh
5
fusion_9*28ˇ´@‡;H¿Obcluster_1_1/xla_runh
9
fusion_33__2*28†∫
@¿0HÄJbcluster_2_1/xla_runh
3
fusion*28˛∑
@‡:H‡Dbcluster_9_1/xla_runh
5
fusion_8*28ﬂ§
@Ä;H†gbcluster_1_1/xla_runh
6
	fusion_10*28ﬂè
@Ä1H‡dbcluster_1_1/xla_runh
6
	fusion_26*28û‘	@ﬂ7HÄIbcluster_1_1/xla_runh
4
copy_57*28ﬂë	@Ä0HÄ=bcluster_0_1/xla_runh
4
copy_57*28ø‰@†.H‡hbcluster_1_1/xla_runh
6
	fusion_18*28›◊@†/H¿=bcluster_1_1/xla_runh
é
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28ù÷@¿$HÄEb%Adam/Adam/update_14/ResourceApplyAdamh
6
	fusion_61*28ﬁæ@Ä0H¿7bcluster_0_1/xla_runh
4
copy_50*28Ä∂@¿*H¿bbcluster_1_1/xla_runh
6
	fusion_16*28†±@‡0H¿6bcluster_2_1/xla_runh
6
	fusion_41*28ø§@‡1H‡4bcluster_1_1/xla_runh
4
add_266*28¿õ@‡*H‡?bcluster_2_1/xla_runh
6
	fusion_43*28ﬂÏ@‡ H‡=bcluster_1_1/xla_runh
6
	fusion_42*28ø„@†H†<bcluster_1_1/xla_runh
6
	fusion_30*28æ„@Ä,H†2bcluster_2_1/xla_runh
3
fusion*28˝’@†'HÄ<bcluster_2_1/xla_runh
5
fusion_6*28øÉ@†)H†-bcluster_4_1/xla_runh
4
copy_72*28øÅ@‡'Hﬂ3bcluster_0_1/xla_runh
5
fusion_6*28ˇÛ@¿"Hü5bcluster_2_1/xla_runh
6
	fusion_33*28†’@¿!H¿1bcluster_2_1/xla_runh
6
	fusion_45*28ø“@Ä"H‡4bcluster_1_1/xla_runh
c
6_ZN5cudnn3cnn23kern_precompute_indicesILb0EEEvPiiiiiii*28†—@¿!HÄ>bcluster_1_1/xla_runh
√
û_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28†©@¿#H†*b
div_no_nanh
6
	fusion_21*28ˇõ@‡ H‡,bcluster_2_1/xla_runh
6
	fusion_50*28ﬂ˛@Ä Hﬂ+bcluster_0_1/xla_runh
6
	fusion_36*28Ä˛@¿ H‡'bcluster_2_1/xla_runh
5
fusion_1*28¿˙@† H†,bcluster_2_1/xla_runh
6
	fusion_44*28¿Ò@¿H¿Ebcluster_1_1/xla_runh
6
	fusion_48*28üÒ@¿ HÄ)bcluster_2_1/xla_runh
9
fusion_33__1*28‡Ó@†H‡'bcluster_2_1/xla_runh
6
	fusion_49*28ø∏@¿H†$bcluster_2_1/xla_runh
4
add_368*28¿∂@ÄH†"bcluster_2_1/xla_runh
¥
Ñ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28¿ê@‡H¿!bAssignAddVariableOp_1h
4
add_331*28†˛@‡HÄ4bcluster_2_1/xla_runh
4
add_343*28¿Ò@ÄH¿'bcluster_2_1/xla_runh
4
add_356*28˝@‡H¿+bcluster_2_1/xla_runh
6
	fusion_42*28øÏ@‡HÄ*bcluster_2_1/xla_runh
3
add_39*28Ä⁄@ÄHÄbcluster_4_1/xla_runh
3
add_11*28¿÷@†HÄ bcluster_7_1/xla_runh
6
	fusion_27*28†“@†HÄ bcluster_2_1/xla_runh
3
fusion*28ﬂŒ@¿H†bcluster_7_1/xla_runh
3
fusion*28ø£@ÄH‡bcluster_5_1/xla_runh
4
slice_1*28ﬂï@‡H¿bcluster_9_1/xla_runh
3
fusion*28˛˝@ÄH‡bcluster_6_1/xla_runh
¥
Ñ_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28‡˘@ÄH†bAssignAddVariableOp_7h
«
£_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28† @† H† b
LogicalAndh