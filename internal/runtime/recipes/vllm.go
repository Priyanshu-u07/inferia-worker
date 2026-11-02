package recipes

import (
	"fmt"
	"strconv"

	"github.com/inferia/inferia-worker/internal/config/vllm"
)

// vllmRecipe builds an invocation of vllm/vllm-openai. It is also reused for
// vllm-omni, which has the same CLI surface.
type vllmRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r vllmRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	// vLLM is GPU-only: enforce at least one index after shared validation.
	if err := requireGPU(in); err != nil {
		return Plan{}, err
	}
	model := stripScheme(in.ArtifactURI)
	cfg := sanitiseConfig(in.Config)

	// ----- GPU-aware default flags -----
	envDefaults := map[string]string{
		"CUDA_MODULE_LOADING": "LAZY",
	}
	gpuCfg, gpuEnv := vllm.GPUOptimalConfig(in.GPUName, in.GPUMemoryMiB, len(in.GPUIndices))
	for k, v := range gpuCfg {
		if _, ok := cfg[k]; !ok {
			cfg[k] = v
		}
	}
	for k, v := range gpuEnv {
		envDefaults[k] = v
	}
	capMaxModelLen(cfg)
	// LD_LIBRARY_PATH is set after gpuEnv so GPU profiles cannot accidentally
	// override it. This path fixes an AWS DLAMI driver mismatch (see below)
	// that would otherwise leave deploys stuck in DEPLOYING.
	//
	// The vllm-openai image bakes a CUDA forward-compat driver (e.g.
	// /usr/local/cuda/compat/libcuda.so.575) ahead of the standard lib dirs
	// in its ld.so cache. CUDA forward-compat is meant to let a newer CUDA
	// userspace run on an OLDER kernel driver. On hosts whose NVIDIA driver
	// is NEWER than that compat lib (e.g. AWS DLAMI driver 580), the loader
	// still binds the compat libcuda — now mismatched against the running
	// kernel module — and CUDA init dies with "Error 803: system has
	// unsupported display driver / cuda driver combination" before a single
	// weight is fetched, leaving the deploy stuck DEPLOYING. Putting the
	// multiarch dir (where the nvidia-container-toolkit injects the HOST
	// driver's libcuda, which always matches the running kernel module)
	// FIRST on LD_LIBRARY_PATH makes the container bind the correct driver
	// and bypass the broken compat shim. The remaining dirs preserve the
	// image's own search path for the rest of the CUDA toolkit.
	envDefaults["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

	cmd := []string{
		model,
		"--served-model-name", model,
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", r.port),
	}
	// Apply config flags. Use --kebab-case to match vLLM convention.
	for _, k := range []string{
		"tensor_parallel_size", "pipeline_parallel_size",
		"dtype", "max_model_len", "max_num_seqs",
		"gpu_memory_utilization", "quantization",
		"kv_cache_dtype", "max_num_batched_tokens",
	} {
		if v, ok := cfg[k]; ok {
			cmd = append(cmd, dashed(k), cliArg(v))
		}
	}
	if v, ok := cfg["enforce_eager"].(bool); ok && v {
		cmd = append(cmd, "--enforce-eager")
	}
	if v, ok := cfg["enable_prefix_caching"].(bool); ok && v {
		cmd = append(cmd, "--enable-prefix-caching")
	}
	if v, ok := cfg["trust_remote_code"].(bool); ok && v {
		cmd = append(cmd, "--trust-remote-code")
	}

	env := mergeEnv(in.Env, envDefaults)

	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-vllm", in.DeploymentID),
		Cmd:           cmd,
		Env:           env,
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}

// capMaxModelLen holds max_model_len, however it was set, to the model's own
// context: vLLM refuses to start above it. Only the control plane can look that
// up, so it arrives in config; unset means unknown.
func capMaxModelLen(cfg map[string]any) {
	native, ok := wholeNumber(cfg["native_max_model_len"])
	if !ok || native <= 0 {
		return
	}
	if current, ok := wholeNumber(cfg["max_model_len"]); ok && current > native {
		cfg["max_model_len"] = native
	}
}

func wholeNumber(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case int32:
		return int(x), true
	case int64:
		return int(x), true
	case float32:
		return int(x), true
	case float64:
		return int(x), true
	case string:
		n, err := strconv.Atoi(x)
		return n, err == nil
	}
	return 0, false
}

func mergeEnv(user, defaults map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range defaults {
		out[k] = v
	}
	for k, v := range user {
		out[k] = v
	}
	return out
}
