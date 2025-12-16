ckpt_dir="data/ckpts/vase_sponge_test1_60hz/ckpts_abs/rdp_ldp/not_change_kernel_size/horizon64"
output_dir="data/outputs/2025.12.15/07.33.30_train_latent_diffusion_unet_image_real_wipe_ldp_vase_sponge1_rdp_ldp_60hz_horizon64"

mv ${output_dir}/logs.json.txt ${ckpt_dir}
mv ${output_dir}/checkpoints/latest.ckpt ${ckpt_dir}