; To be used with NASM-Compatible Assemblers

; System V AMD64 ABI Convention

; Function parameters are passed this way:
; Interger values: RDI, RSI, RDX, RCX, R8, R9, R10, and R11
; Float Point values: XMM0, XMM1, XMM2, XMM3 

; Function return value is returned this way:
; Integer value: RAX:RDX 
; Float Point value: XMM0

; SSE, SSE2

; Luis Delgado. November 13, 2018 (Date of last edition).

; These functions were written to process audio data.


;********************************
; .data
;********************************

section .data
BaseVolumeFixedPoint16Bits_32bitsfloat equ 0x47000000




;********************************
; CODE
;********************************

section .text

global GenerateVolumeforFixedPoint16bitsStereo
;void GenerateVolumeforFixedPoint16bitsStereo(float Normalized_Volume, uint_32t * Generated_Volume)
;*********************************************************************
;Given the normalized in IEEE-756 format,
;This format returns 2 copies of a 16-bits Fixed-Point value
;This value shall be used in a interleaved Stereo 16-bits PCM buffer.
;*********************************************************************
GenerateVolumeforFixedPoint16bitsStereo:
    enter 0,0
    push dword BaseVolumeFixedPoint16Bits_32bitsfloat
	fld dword [rsp]
	    movss [rsp],xmm0
        fmul dword [rsp]
        fistp dword [rdi]
    mov dx, word [rdi]
    ;mov dword [rsi], 0xAAAAAAAA
    mov [rdi+2],dx
    add rsp,8
    leave
    ret

global ChangeVolume16bits
;void ChangeVolume16bits(&Dest, &Orig, uint64_t Bytes, uint32_t Volume);
;*********************************************************************
;This function change the volume of a 16-bits sample buffer
;The volume used shall be two contiguos 16-bits volume values
;The volume can the generated using GenerateVolumeforFixedPoint16bitsStereo
;The size of the buffer should be a multiple of 16
;*********************************************************************
ChangeVolume16bits:
    enter 0,0

    push rcx

    movss xmm1, [rsp]
    shufps xmm1,xmm1,0
    add rsp,8
    push rax
    mov  rcx,16
    mov rax, rdx
    xor rdx,rdx
    div rcx
    mov rcx,rax

    cicloxmmecx:
        mov rax,rcx
        shl rax,4
        sub rax,16
                movups xmm0,[rsi+rax]
                pmulhw xmm0,xmm1
                psllw xmm0,1
                movups [rdi+rax],xmm0
        loop cicloxmmecx

    pop rax
    leave
    ret
