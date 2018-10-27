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

; Collection of varied simple functions.



;********************************
; CODE
;********************************

section .text
global fastmemcpy; void fastmemcpy(void * Destiny, void * Source, uint64_t bytes)
;***************************************************************
; It uses the XMM registers to move data from Source to Destiny
;***************************************************************
fastmemcpy:
    enter 0,0

    mov rcx,rdx

    Loopprincipal:
        cmp rcx,16
        jb NohacerXMM128
            movups xmm0,[rsi+rcx-16]
            movups [rdi+rcx-16],xmm0
            sub rcx,15
            jmp TerminarCiclo
        NohacerXMM128:
            cmp rcx,8
            jb NohacerXMM64
                movsd xmm0,[rsi+rcx-8]
                movsd [rdi+rcx-8],xmm0
            sub rcx,7
            jmp TerminarCiclo
        NohacerXMM64:
            cmp rcx,4
            jb NohacerREG32
                mov edx, dword [rsi+rcx-4]
                mov [rdi+rcx-4],edx
            sub rcx,3
            jmp TerminarCiclo
        NohacerREG32:
            cmp rcx,2
            jb NohacerREG16
                mov dx, word [rsi+rcx-2]
                mov [rdi+rcx-2],dx
            sub rcx,1
            jmp TerminarCiclo
        NohacerREG16:
            mov dl, byte [rsi+rcx-1]
            mov [rdi+rcx-1],dl
        TerminarCiclo:
        loop Loopprincipal
    leave
    ret


global CheckBit; char CheckBit(uint32_t INT32,uint8_t BIT);
;***************************************************************
; It checks if a bit is set in a 32-bits value
;***************************************************************
CheckBit:
    enter 0,0
    mov rcx,rsi
    shr edi,cl
    and edi,1
    movzx rax,di
    leave
    ret

global Buffer_16bits_ADD; 
;void Buffer_16bits_ADD(uint16_t * A_Result, uint16_t * B, uint64_t bytes)
;**************************************************************************
; It takes two buffers and sums its values in groups of 16-bits (short int)
;**************************************************************************
Buffer_16bits_ADD:
    enter 0,0
    mov rcx,rdx

    LoopPrincipal:
            cmp rcx,16
            jb SumarEn64
                movups xmm0,[rsi+rcx-16]
                movups xmm1,[rdi+rcx-16]
                paddsw xmm0,xmm1
                movups [rdi+rcx-16],xmm0
            sub rcx,15
	    jmp TerminaLoopDeSumas16

	SumarEn64:
	    cmp rcx,8
	    jb SumarEn32
	        movsd xmm0,[rsi+rcx-8]
                movsd xmm1,[rdi+rcx-8]
                paddsw xmm0,xmm1
                movsd [rdi+rcx-8],xmm0
	    sub rcx,7
	    jmp TerminaLoopDeSumas16

	SumarEn32:
	    cmp rcx,4
	    jb SumarEn16
	        movss xmm0,[rsi+rcx-4]
                movss xmm1,[rdi+rcx-4]
                paddsw xmm0,xmm1
                movss [rdi+rcx-4],xmm0
	    sub rcx,3
	    jmp TerminaLoopDeSumas16

	SumarEn16:
	    cmp rcx,2
	    jb TerminaLoopDeSumas16
		mov dx,[rsi+rcx-2]
                add dx,[rdi+rcx-2]
                mov  [rdi+rcx-2],dx

	TerminaLoopDeSumas16:
        loop LoopPrincipal
    salida:

    leave
    ret


global Buffer_Clear; void ClearBuffer(void * Buffer_ptr,uint64_t Bytes)
;**********************************************
; It clears (fills with zeroes) a buffer
;**********************************************
    Buffer_Clear:
    enter 0,0

    pxor xmm0,xmm0
    xor rdx,rdx
    mov rcx,rsi

    CBML:
	cmp rcx,16
	jb CB8
	    movups [rdi+rcx-16],xmm0
	    sub rcx,15
	jmp CBEND

    CB8:
	cmp rcx,8
	jb CB4
	    mov [rdi+rcx-8], rdx
	    sub rcx,7
	jmp CBEND

    CB4:
	cmp rcx,4
	jb CB2
	    mov [rdi+rcx-4], edx
	    sub rcx,3
	jmp CBEND
    CB2:
	cmp rcx,2
	jb CB1
	    mov [rdi+rcx-2], dx
	    sub rcx,1
	jmp CBEND
    CB1:
	mov [rdi],dl

    CBEND:
    loop CBML

    leave
    ret
