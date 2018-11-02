; To be used with NASM-Compatible Assemblers

; System V AMD64 ABI Convention (*nix)
; Function parameters are passed this way:
; Interger values: RDI, RSI, RDX, RCX, R8, R9
; Float Point values: arg1f, arg2f, arg3f, RCXf, XMM4, XMM5, XMM6, XMM7
; Extra arguments are pushed on the stack starting from the most left argument
; Function return value is returned this way:
; Integer value: RAX:RDX 
; Float Point value: arg1f:arg2f
; RAX, R10 and R11 are volatile

; Microsoft x64 calling convention (Windows)
; Function parameters are passed this way:
; Interger values: RCX, RDX, R8, R9
; Float Point values: arg1f, arg2f, arg3f, RCXf 
; Both kind of arguments are counted together
; e.g. if the second argument is a float, it will be in arg2f, if Interger, then RDX
; Extra arguments are pushed on the stack 
; Function return value is returned this way:
; Integer value: RAX
; Float Point value: arg1f
; XMM4, XMM5, RAX, R10 and R11 are volatile

; SSE, SSE2, SSE3

; Luis Delgado.

;November 13, 2018: Date of last edition (before translation).
;Octuber 2, 2018:   Adding Windows support.

; Collection of varied simple functions.


;*****************************
;MACROS
;*****************************

%ifidn __OUTPUT_FORMAT__, elf64 
%elifidn __OUTPUT_FORMAT__, win64
%endif

%ifidn __OUTPUT_FORMAT__, win64 
%endif

%macro args_reset 0
    %ifidn __OUTPUT_FORMAT__, elf64 
        %define arg1 RDI
        %define arg2 RSI
        %define arg3 RDX
        %define RCX RCX
        %define arg5 R8
        %define arg6 R9 
    %elifidn __OUTPUT_FORMAT__, win64
        %define arg1 RCX
        %define arg2 RDX
        %define arg3 R8
        %define RCX R9
        %define arg5 [rbp+48]
        %define arg6 [rbp+48+8]
    %endif

    %define arg1f XMM0
    %define arg2f XMM1
    %define arg3f XMM2
    %define RCXf XMM3
%endmacro

args_reset ;<--Sets arguments definitions to normal, as it's definitions can change.

;********************************
; CODE
;********************************

section .text
global fastmemcpy; void fastmemcpy(void * Destiny, void * Source, uint64_t bytes)
;***************************************************************
; It uses the XMM registers to move data from Source to Destiny
;***************************************************************
fastmemcpy:
%ifidn __OUTPUT_FORMAT__, win64 
    mov RAX,RCX
    %define arg1 RAX
%endif

    enter 0,0
    mov RCX,arg3

    Loopprincipal:
        cmp RCX,16
        jb NohacerXMM128
            movups xmm0,[arg2+RCX-16]
            movups [arg1+RCX-16],xmm0
            sub RCX,15
            jmp TerminarCiclo
        NohacerXMM128:
            cmp RCX,8
            jb NohacerXMM64
                movsd xmm0,[arg2+RCX-8]
                movsd [arg1+RCX-8],xmm0
            sub RCX,7
            jmp TerminarCiclo
        NohacerXMM64:
            cmp RCX,4
            jb NohacerREG32
                mov edx, dword [arg2+RCX-4]
                mov [arg1+RCX-4],edx
            sub RCX,3
            jmp TerminarCiclo
        NohacerREG32:
            cmp RCX,2
            jb NohacerREG16
                mov dx, word [arg2+RCX-2]
                mov [arg1+RCX-2],dx
            sub RCX,1
            jmp TerminarCiclo
        NohacerREG16:
            mov dl, byte [arg2+RCX-1]
            mov [arg1+RCX-1],dl
        TerminarCiclo:
        loop Loopprincipal
    leave

%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif

    ret



global CheckBit; char CheckBit(uint32_t INT32,uint8_t BIT);
;***************************************************************
; It checks if a bit is set in a 32-bits value
;***************************************************************
CheckBit:
%ifidn __OUTPUT_FORMAT__, win64 
    mov rax,rcx
%endif
    enter 0,0
    mov RCX,arg2
    shr rax,cl
    and rax,1
    leave
%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif
    ret


global Buffer_16bits_ADD; 
;void Buffer_16bits_ADD(uint16_t * A_Result, uint16_t * B, uint64_t bytes)
;**************************************************************************
; It takes two buffers and sums its values in groups of 16-bits (short int)
;**************************************************************************
Buffer_16bits_ADD:
    enter 0,0

%ifidn __OUTPUT_FORMAT__, win64 
    mov RAX,RCX
    %define arg1 RAX
%endif

    mov RCX,arg3

    LoopPrincipal:
            cmp RCX,16
            jb SumarEn64
                movups xmm0,[arg2+RCX-16]
                movups xmm1,[arg1+RCX-16]
                paddsw xmm0,xmm1
                movups [arg1+RCX-16],xmm0
            sub RCX,15
	    jmp TerminaLoopDeSumas16

	SumarEn64:
	    cmp RCX,8
	    jb SumarEn32
	        movsd xmm0,[arg2+RCX-8]
                movsd xmm1,[arg1+RCX-8]
                paddsw xmm0,xmm1
                movsd [arg1+RCX-8],xmm0
	    sub RCX,7
	    jmp TerminaLoopDeSumas16

	SumarEn32:
	    cmp RCX,4
	    jb SumarEn16
	        movss xmm0,[arg2+RCX-4]
                movss xmm1,[arg1+RCX-4]
                paddsw xmm0,xmm1
                movss [arg1+RCX-4],xmm0
	    sub RCX,3
	    jmp TerminaLoopDeSumas16

	SumarEn16:
	    cmp RCX,2
	    jb TerminaLoopDeSumas16
		mov dx,[arg2+RCX-2]
                add dx,[arg1+RCX-2]
                mov  [arg1+RCX-2],dx

	TerminaLoopDeSumas16:
        loop LoopPrincipal
    salida:

%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif

    leave
    ret


global Buffer_Clear; void ClearBuffer(void * Buffer_ptr,uint64_t Bytes)
;**********************************************
; It clears (fills with zeroes) a buffer
;**********************************************
    Buffer_Clear:

%ifidn __OUTPUT_FORMAT__, win64 
    mov RAX,RCX
    %define arg1 RAX
%endif

    enter 0,0

    pxor xmm0,xmm0
    xor arg3,arg3
    mov RCX,arg2

    CBML:
	cmp RCX,16
	jb CB8
	    movups [arg1+RCX-16],xmm0
	    sub RCX,15
	jmp CBEND

    CB8:
	cmp RCX,8
	jb CB4
	    mov [arg1+RCX-8], arg3
	    sub RCX,7
	jmp CBEND

    CB4:
	cmp RCX,4
	jb CB2
	    mov [arg1+RCX-4], edx
	    sub RCX,3
	jmp CBEND
    CB2:
	cmp RCX,2
	jb CB1
	    mov [arg1+RCX-2], dx
	    sub RCX,1
	jmp CBEND
    CB1:
	mov [arg1],dl

    CBEND:
    loop CBML

%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif

    leave
    ret
