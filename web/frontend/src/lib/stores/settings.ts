import { writable } from 'svelte/store';
import type { InferenceParams, OutputConfig } from '$lib/api';

/**
 * Create a writable store that persists to localStorage.
 * Falls back to the default value if localStorage is unavailable or empty.
 */
function persisted<T>(key: string, defaultValue: T) {
	let initial = defaultValue;
	if (typeof window !== 'undefined') {
		try {
			const stored = localStorage.getItem(key);
			if (stored !== null) {
				initial = JSON.parse(stored);
			}
		} catch {
			// ignore parse errors
		}
	}

	const store = writable<T>(initial);

	store.subscribe((value) => {
		if (typeof window !== 'undefined') {
			try {
				localStorage.setItem(key, JSON.stringify(value));
			} catch {
				// ignore quota errors
			}
		}
	});

	return store;
}

export const autoExtractFrames = persisted<boolean>('ck:autoExtractFrames', true);

export const defaultParams = persisted<InferenceParams>('ck:defaultParams', {
	input_is_linear: false,
	despill_strength: 1.0,
	auto_despeckle: true,
	despeckle_size: 400,
	refiner_scale: 1.0
});

export const defaultOutputConfig = persisted<OutputConfig>('ck:defaultOutputConfig', {
	fg_enabled: true,
	fg_format: 'exr',
	matte_enabled: true,
	matte_format: 'exr',
	comp_enabled: true,
	comp_format: 'png',
	processed_enabled: true,
	processed_format: 'exr'
});
