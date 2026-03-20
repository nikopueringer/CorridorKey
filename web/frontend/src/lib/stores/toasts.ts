import { writable } from 'svelte/store';

export interface Toast {
	id: number;
	message: string;
	type: 'info' | 'success' | 'error' | 'warning';
	duration: number;
}

let nextId = 0;

export const toasts = writable<Toast[]>([]);

export function addToast(message: string, type: Toast['type'] = 'info', duration = 4000) {
	const id = nextId++;
	toasts.update((t) => [...t, { id, message, type, duration }]);
	if (duration > 0) {
		setTimeout(() => removeToast(id), duration);
	}
}

export function removeToast(id: number) {
	toasts.update((t) => t.filter((toast) => toast.id !== id));
}

// Convenience aliases
export const toast = {
	info: (msg: string, duration?: number) => addToast(msg, 'info', duration),
	success: (msg: string, duration?: number) => addToast(msg, 'success', duration),
	error: (msg: string, duration?: number) => addToast(msg, 'error', duration ?? 6000),
	warning: (msg: string, duration?: number) => addToast(msg, 'warning', duration),
};
