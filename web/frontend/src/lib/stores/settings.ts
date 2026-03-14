import { writable } from 'svelte/store';
import type { InferenceParams, OutputConfig } from '$lib/api';

export const autoExtractFrames = writable<boolean>(true);

export const defaultParams = writable<InferenceParams>({
	input_is_linear: false,
	despill_strength: 1.0,
	auto_despeckle: true,
	despeckle_size: 400,
	refiner_scale: 1.0
});

export const defaultOutputConfig = writable<OutputConfig>({
	fg_enabled: true,
	fg_format: 'exr',
	matte_enabled: true,
	matte_format: 'exr',
	comp_enabled: true,
	comp_format: 'png',
	processed_enabled: true,
	processed_format: 'exr'
});
