import { writable } from 'svelte/store';
import type { VRAMInfo } from '$lib/api';
import { api } from '$lib/api';

export const device = writable<string>('detecting...');
export const vram = writable<VRAMInfo | null>(null);
export const wsConnected = writable(false);

export async function refreshDevice() {
	try {
		const res = await api.system.device();
		device.set(res.device);
	} catch {
		device.set('unknown');
	}
}

export async function refreshVRAM() {
	try {
		const res = await api.system.vram();
		if (res.available) {
			vram.set(res);
		}
	} catch {
		// ignore
	}
}
