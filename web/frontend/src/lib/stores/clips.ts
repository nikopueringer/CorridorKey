import { writable } from 'svelte/store';
import type { Clip } from '$lib/api';
import { api } from '$lib/api';

export const clips = writable<Clip[]>([]);
export const clipsDir = writable<string>('');
export const clipsLoading = writable(false);
export const clipsError = writable<string | null>(null);

export async function refreshClips() {
	clipsLoading.set(true);
	clipsError.set(null);
	try {
		const res = await api.clips.list();
		clips.set(res.clips);
		clipsDir.set(res.clips_dir);
	} catch (e) {
		clipsError.set(e instanceof Error ? e.message : String(e));
	} finally {
		clipsLoading.set(false);
	}
}
