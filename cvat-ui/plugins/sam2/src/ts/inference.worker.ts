// Copyright (C) 2024 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { InferenceSession, env, Tensor } from 'onnxruntime-web';

let decoder: InferenceSession | null = null;

// WASM path 설정
env.wasm.wasmPaths = '/assets/';

// Worker에서 처리할 Action 타입 정의
export enum WorkerAction {
    INIT = 'init',
    DECODE = 'decode',
}

// INIT에 필요한 값
export interface InitBody {
    decoderURL: string;
}

// DECODE에 필요한 값
export interface DecodeBody {
    image_embed: Tensor;
    high_res_feats_0: Tensor;
    high_res_feats_1: Tensor;
    point_coords: Tensor;
    point_labels: Tensor;
    orig_im_size: Tensor;
    mask_input: Tensor;
    has_mask_input: Tensor;
    readonly [name: string]: Tensor;
}

// 메시지 타입 정의
export interface WorkerOutput {
    action: WorkerAction;
    error?: string;
    payload?: any;
}

export interface WorkerInput {
    action: WorkerAction;
    payload: InitBody | DecodeBody;
}

// 에러를 string으로 변환하는 함수
const errorToMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return `${error.message}\n${error.stack}`;
    }
    if (typeof error === 'string') {
        return error;
    }
    return 'Unknown error, check browser console';
};

// WebWorker 실행
if ((self as any).importScripts) {
    console.log('[Worker] Initialized and listening for messages');

    onmessage = async (e: MessageEvent<WorkerInput>) => {
        const { action, payload } = e.data;

        try {
            if (action === WorkerAction.INIT) {
                if (decoder) {
                    console.log('[Worker] Decoder already initialized');
                    return;
                }

                const { decoderURL } = payload as InitBody;
                console.log(`[Worker] Initializing decoder from URL: ${decoderURL}`);

                decoder = await InferenceSession.create(decoderURL);
                console.log('[Worker] Decoder initialized');

                postMessage({ action }); // 성공 알림

            } else if (action === WorkerAction.DECODE) {
                if (!decoder) {
                    const msg = '[Worker] Decoder not initialized';
                    console.error(msg);
                    postMessage({ action, error: msg });
                    return;
                }

                const decodePayload = payload as DecodeBody;
                console.log('[Worker] Running decoder...');

                const results = await decoder.run(decodePayload);
                console.log('[Worker] Decoder run completed');

                postMessage({
                    action,
                    payload: {
                        masks: results.masks,
                        lowResMasks: results.low_res_masks,
                        xtl: Number(results.xtl.data[0]),
                        ytl: Number(results.ytl.data[0]),
                        xbr: Number(results.xbr.data[0]),
                        ybr: Number(results.ybr.data[0]),
                    },
                });
            }
        } catch (err: unknown) {
            const message = errorToMessage(err);
            console.error('[Worker] Error occurred:', message);
            postMessage({ action, error: message });
        }
    };
}

