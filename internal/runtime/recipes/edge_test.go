package recipes

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCLIArg_AllScalarTypes(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{"abc", "abc"},
		{true, "true"},
		{false, "false"},
		{42, "42"},
		{int32(7), "7"},
		{int64(99), "99"},
		{float32(0.5), "0.5"},
		{float64(0.9), "0.9"},
		{struct{ N int }{N: 1}, "{1}"}, // fallback
	}
	for _, tc := range cases {
		if got := cliArg(tc.in); got != tc.want {
			t.Errorf("cliArg(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestStripScheme_NoSchemePassthrough(t *testing.T) {
	if got := stripScheme("no-scheme"); got != "no-scheme" {
		t.Errorf("got %q", got)
	}
	if got := stripScheme("hf://org/m"); got != "org/m" {
		t.Errorf("got %q", got)
	}
}

func TestVLLM_EnforceEagerAndTrustRemoteCode(t *testing.T) {
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://o/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config: map[string]any{
			"enforce_eager":     true,
			"trust_remote_code": true,
		},
	})
	if err != nil {
		t.Fatalf("%v", err)
	}
	joined := strings.Join(plan.Cmd, " ")
	if !strings.Contains(joined, "--enforce-eager") {
		t.Errorf("missing --enforce-eager: %v", plan.Cmd)
	}
	if !strings.Contains(joined, "--trust-remote-code") {
		t.Errorf("missing --trust-remote-code: %v", plan.Cmd)
	}
}

func TestVLLM_BoolFalseDoesNotAddFlag(t *testing.T) {
	r, _ := Get("vllm")
	plan, _ := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://o/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config:       map[string]any{"enforce_eager": false, "trust_remote_code": false},
	})
	joined := strings.Join(plan.Cmd, " ")
	if strings.Contains(joined, "--enforce-eager") || strings.Contains(joined, "--trust-remote-code") {
		t.Errorf("bool=false should not emit flag: %v", plan.Cmd)
	}
}

func TestInfinity_BatchAndDtype(t *testing.T) {
	r, _ := Get("infinity")
	plan, _ := r.BuildPlan(BuildInput{
		DeploymentID: "d",
		ArtifactURI:  "hf://o/m",
		GPUIndices:   []int{0},
		HostPort:     1234,
		Config:       map[string]any{"max_batch_size": 32, "dtype": "float16"},
	})
	joined := strings.Join(plan.Cmd, " ")
	if !strings.Contains(joined, "--batch-size 32") {
		t.Errorf("missing batch-size: %v", plan.Cmd)
	}
	if !strings.Contains(joined, "--dtype float16") {
		t.Errorf("missing dtype: %v", plan.Cmd)
	}
}

// Verify validate() rejects propagate from each recipe (cover BuildPlan's err return).
func TestAllRecipes_RejectBadInput(t *testing.T) {
	bad := BuildInput{} // empty: no DeploymentID, no GPUs, no URI
	for _, name := range Names() {
		r, _ := Get(name)
		if _, err := r.BuildPlan(bad); err == nil {
			t.Errorf("%s: expected validate error", name)
		}
	}
}

func TestContainerName_Sanitises(t *testing.T) {
	got := containerName("p", "abc/def:ghi..jkl")
	if got != "p-abc-def-ghi-jkl" {
		t.Errorf("got %q", got)
	}
}

// The control plane sends config as JSON, so every number arrives as a float64.
// These are the settings from the dashboard's New Deployment form.
func TestVLLM_DashboardSettingsBecomeFlags(t *testing.T) {
	var config map[string]any
	if err := json.Unmarshal([]byte(`{
		"max_model_len": 8192, "gpu_memory_utilization": 0.8, "dtype": "bfloat16",
		"quantization": "fp8", "enforce_eager": true, "max_num_seqs": 64,
		"kv_cache_dtype": "fp8", "trust_remote_code": true
	}`), &config); err != nil {
		t.Fatal(err)
	}
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep-settings",
		ArtifactURI:  "hf://Qwen/Qwen3-0.6B",
		Config:       config,
		GPUIndices:   []int{0},
		HostPort:     19000,
	})
	if err != nil {
		t.Fatal(err)
	}
	cmd := strings.Join(plan.Cmd, " ")
	for _, want := range []string{
		"--max-model-len 8192 ", "--gpu-memory-utilization 0.8 ", "--dtype bfloat16 ",
		"--quantization fp8 ", "--max-num-seqs 64 ", "--kv-cache-dtype fp8 ",
		"--enforce-eager", "--trust-remote-code",
	} {
		if !strings.Contains(cmd+" ", want) {
			t.Errorf("missing %q in: %s", want, cmd)
		}
	}
}

func vllmCmd(t *testing.T, gpu string, memoryMiB uint64, config map[string]any) []string {
	t.Helper()
	r, _ := Get("vllm")
	plan, err := r.BuildPlan(BuildInput{
		DeploymentID: "dep", ArtifactURI: "hf://o/m", Config: config,
		GPUIndices: []int{0}, HostPort: 1234, GPUName: gpu, GPUMemoryMiB: memoryMiB,
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan.Cmd
}

func flagValue(cmd []string, flag string) string {
	for i, tok := range cmd {
		if tok == flag && i+1 < len(cmd) {
			return cmd[i+1]
		}
	}
	return ""
}

func TestVLLM_MaxModelLenIsHeldToTheModelsContext(t *testing.T) {
	// A blank form on an H100 defaults to 65536; a small model stops vLLM starting.
	cmd := vllmCmd(t, "NVIDIA H100", 81920, map[string]any{"native_max_model_len": float64(2048)})
	if got := flagValue(cmd, "--max-model-len"); got != "2048" {
		t.Errorf("profile default not capped: --max-model-len %q", got)
	}
	cmd = vllmCmd(t, "NVIDIA A10G", 24576, map[string]any{
		"native_max_model_len": float64(2048), "max_model_len": float64(8192)})
	if got := flagValue(cmd, "--max-model-len"); got != "2048" {
		t.Errorf("deployment value not capped: --max-model-len %q", got)
	}
}

func TestVLLM_MaxModelLenWithinOrUnknownContextIsKept(t *testing.T) {
	cmd := vllmCmd(t, "NVIDIA A10G", 24576, map[string]any{"native_max_model_len": float64(131072)})
	if got := flagValue(cmd, "--max-model-len"); got != "8192" {
		t.Errorf("--max-model-len %q, want the profile's 8192", got)
	}
	cmd = vllmCmd(t, "NVIDIA A10G", 24576, nil)
	if got := flagValue(cmd, "--max-model-len"); got != "8192" {
		t.Errorf("--max-model-len %q, want the profile's 8192", got)
	}
}

func TestVLLM_ControlPlaneLimitIsNotAFlag(t *testing.T) {
	cmd := vllmCmd(t, "", 0, map[string]any{"native_max_model_len": float64(2048)})
	if strings.Contains(strings.Join(cmd, " "), "native") {
		t.Errorf("internal key became a flag: %v", cmd)
	}
}

func TestVLLM_OtherEnginesKnobsAreNotFlags(t *testing.T) {
	// vLLM has none of these, and exits on an unknown flag.
	cmd := strings.Join(vllmCmd(t, "", 0, map[string]any{
		"max_batch_size": 8, "max_input_length": 1024, "max_total_tokens": 2048,
	}), " ")
	for _, flag := range []string{"--max-batch-size", "--max-input-length", "--max-total-tokens"} {
		if strings.Contains(cmd, flag) {
			t.Errorf("%s passed to vLLM: %s", flag, cmd)
		}
	}
}

func TestVLLM_QuantizationComesOnlyFromTheDeployment(t *testing.T) {
	if got := flagValue(vllmCmd(t, "NVIDIA H100", 81920, nil), "--quantization"); got != "" {
		t.Errorf("profile added --quantization %q", got)
	}
	cmd := vllmCmd(t, "NVIDIA H100", 81920, map[string]any{"quantization": "awq"})
	if got := flagValue(cmd, "--quantization"); got != "awq" {
		t.Errorf("--quantization %q, want awq", got)
	}
}
