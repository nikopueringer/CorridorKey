/** Typed fetch wrappers for the CorridorKey API. */

const BASE = '';

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
	const opts: RequestInit = {
		method,
		headers: { 'Content-Type': 'application/json' }
	};
	if (body !== undefined) {
		opts.body = JSON.stringify(body);
	}
	const res = await fetch(`${BASE}${path}`, opts);
	if (!res.ok) {
		const detail = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(detail.detail || res.statusText);
	}
	return res.json();
}

// --- Types ---

export interface ClipAsset {
	path: string;
	asset_type: string;
	frame_count: number;
}

export interface Clip {
	name: string;
	root_path: string;
	state: string;
	input_asset: ClipAsset | null;
	alpha_asset: ClipAsset | null;
	mask_asset: ClipAsset | null;
	frame_count: number;
	completed_frames: number;
	has_outputs: boolean;
	warnings: string[];
	error_message: string | null;
}

export interface ClipListResponse {
	clips: Clip[];
	clips_dir: string;
}

export interface Job {
	id: string;
	job_type: string;
	clip_name: string;
	status: string;
	current_frame: number;
	total_frames: number;
	error_message: string | null;
}

export interface JobListResponse {
	current: Job | null;
	queued: Job[];
	history: Job[];
}

export interface InferenceParams {
	input_is_linear: boolean;
	despill_strength: number;
	auto_despeckle: boolean;
	despeckle_size: number;
	refiner_scale: number;
}

export interface OutputConfig {
	fg_enabled: boolean;
	fg_format: string;
	matte_enabled: boolean;
	matte_format: string;
	comp_enabled: boolean;
	comp_format: string;
	processed_enabled: boolean;
	processed_format: string;
}

export interface VRAMInfo {
	total: number;
	reserved: number;
	allocated: number;
	free: number;
	name: string;
	available: boolean;
}

export interface Project {
	name: string;
	display_name: string;
	path: string;
	clip_count: number;
	created: string | null;
	clips: Clip[];
}

export interface DeviceInfo {
	device: string;
}

export interface WeightInfo {
	installed: boolean;
	path: string;
	detail: string | null;
	size_hint: string;
	download?: { status: string; error: string | null };
}

// --- API calls ---

export const api = {
	projects: {
		list: () => request<Project[]>('GET', '/api/projects'),
		create: (name: string) => request<Project>('POST', '/api/projects', { name }),
		rename: (name: string, display_name: string) =>
			request<unknown>('PATCH', `/api/projects/${encodeURIComponent(name)}`, { display_name }),
		delete: (name: string) => request<unknown>('DELETE', `/api/projects/${encodeURIComponent(name)}`)
	},
	clips: {
		list: () => request<ClipListResponse>('GET', '/api/clips'),
		get: (name: string) => request<Clip>('GET', `/api/clips/${encodeURIComponent(name)}`),
		delete: (name: string) => request<unknown>('DELETE', `/api/clips/${encodeURIComponent(name)}`),
		move: (name: string, targetProject: string) =>
			request<unknown>('POST', `/api/clips/${encodeURIComponent(name)}/move?target_project=${encodeURIComponent(targetProject)}`)
	},
	jobs: {
		list: () => request<JobListResponse>('GET', '/api/jobs'),
		submitInference: (
			clip_names: string[],
			params?: Partial<InferenceParams>,
			output_config?: Partial<OutputConfig>,
			frame_range?: [number, number] | null
		) =>
			request<Job[]>('POST', '/api/jobs/inference', {
				clip_names,
				params: params ?? {},
				output_config: output_config ?? {},
				frame_range: frame_range ?? null
			}),
		submitPipeline: (
			clip_names: string[],
			alpha_method = 'gvm',
			params?: Partial<InferenceParams>,
			output_config?: Partial<OutputConfig>
		) =>
			request<Job[]>('POST', '/api/jobs/pipeline', {
				clip_names,
				alpha_method,
				params: params ?? {},
				output_config: output_config ?? {}
			}),
		submitExtract: (clip_names: string[]) =>
			request<Job[]>('POST', '/api/jobs/extract', { clip_names }),
		submitGVM: (clip_names: string[]) =>
			request<Job[]>('POST', '/api/jobs/gvm', { clip_names }),
		submitVideoMaMa: (clip_names: string[], chunk_size = 50) =>
			request<Job[]>('POST', '/api/jobs/videomama', { clip_names, chunk_size }),
		getLog: (jobId: string) => request<Record<string, unknown>>('GET', `/api/jobs/${jobId}/log`),
		cancel: (jobId: string) => request<unknown>('DELETE', `/api/jobs/${jobId}`),
		cancelAll: () => request<unknown>('DELETE', '/api/jobs')
	},
	system: {
		device: () => request<DeviceInfo>('GET', '/api/system/device'),
		vram: () => request<VRAMInfo>('GET', '/api/system/vram'),
		unload: () => request<unknown>('POST', '/api/system/unload'),
		getVramLimit: () => request<{ vram_limit_gb: number }>('GET', '/api/system/vram-limit'),
		setVramLimit: (gb: number) => request<unknown>('POST', `/api/system/vram-limit?vram_limit_gb=${gb}`),
		weights: () => request<Record<string, WeightInfo>>('GET', '/api/system/weights'),
		downloadWeights: (name: string) => request<unknown>('POST', `/api/system/weights/download/${name}`)
	},
	upload: {
		video: async (file: File, name?: string, autoExtract = true): Promise<{ status: string; clips: Clip[]; extract_jobs: string[] }> => {
			const form = new FormData();
			form.append('file', file);
			const qs = new URLSearchParams();
			if (name) qs.set('name', name);
			qs.set('auto_extract', String(autoExtract));
			const res = await fetch(`${BASE}/api/upload/video?${qs}`, { method: 'POST', body: form });
			if (!res.ok) {
				const detail = await res.json().catch(() => ({ detail: res.statusText }));
				throw new Error(detail.detail || res.statusText);
			}
			return res.json();
		},
		frames: async (file: File, name?: string): Promise<{ status: string; clips: Clip[]; frame_count: number }> => {
			const form = new FormData();
			form.append('file', file);
			const params = name ? `?name=${encodeURIComponent(name)}` : '';
			const res = await fetch(`${BASE}/api/upload/frames${params}`, { method: 'POST', body: form });
			if (!res.ok) {
				const detail = await res.json().catch(() => ({ detail: res.statusText }));
				throw new Error(detail.detail || res.statusText);
			}
			return res.json();
		},
		mask: async (clipName: string, file: File): Promise<unknown> => {
			const form = new FormData();
			form.append('file', file);
			const res = await fetch(`${BASE}/api/upload/mask/${encodeURIComponent(clipName)}`, { method: 'POST', body: form });
			if (!res.ok) {
				const detail = await res.json().catch(() => ({ detail: res.statusText }));
				throw new Error(detail.detail || res.statusText);
			}
			return res.json();
		},
		alpha: async (clipName: string, file: File): Promise<unknown> => {
			const form = new FormData();
			form.append('file', file);
			const res = await fetch(`${BASE}/api/upload/alpha/${encodeURIComponent(clipName)}`, { method: 'POST', body: form });
			if (!res.ok) {
				const detail = await res.json().catch(() => ({ detail: res.statusText }));
				throw new Error(detail.detail || res.statusText);
			}
			return res.json();
		}
	},
	preview: {
		url: (clipName: string, passName: string, frame: number) =>
			`${BASE}/api/preview/${encodeURIComponent(clipName)}/${passName}/${frame}`
	}
};
