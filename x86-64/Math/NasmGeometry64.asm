; To be used with NASM-Compatible Assemblers

; System V AMD64 ABI Convention (*nix)
; Function parameters are passed this way:
; Interger values: RDI, RSI, RDX, RCX, R8, R9
; Float Point values: XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7
; Extra arguments are pushed on the stack starting from the most left argument
; Function return value is returned this way:
; Integer value: RAX:RDX 
; Float Point value: XMM0:XMM1
; RAX, R10 and R11 are volatile

; Microsoft x64 calling convention (Windows)
; Function parameters are passed this way:
; Interger values: RCX, RDX, R8, R9
; Float Point values: XMM0, XMM1, XMM2, XMM3 
; Both kind of arguments are counted together
; e.g. if the second argument is a float, it will be in arg2f, if Interger, then RDX
; Extra arguments are pushed on the stack 
; Function return value is returned this way:
; Integer value: RAX
; Float Point value: XMM0
; XMM4, XMM5, RAX, R10 and R11 are volatile

; SSE, SSE2, SSE3

; December 12th, 2018:	First version

; Luis Delgado.


; Collection of mathematical functions, very useful for geometry.


%ifndef _NASMGEOMETRY_ASM_
%define _NASMGEOMETRY_ASM_

%include "NasmMath.asm"

;*****************************
;MACROS
;*****************************

%ifidn __OUTPUT_FORMAT__, elf64 
%elifidn __OUTPUT_FORMAT__, win64
%endif

%ifidn __OUTPUT_FORMAT__, win64 
%endif

args_reset ;<--Sets arguments definitions to normal, as it's definitions can change.

;********************************************************************
; .data
;********************************************************************

fc_3f_mem: dd 01000000010000000000000000000000b ; 3.0f;

;********************************
; CODE
;********************************

section .text

global Triangle_3D_Baricenter; Triangle_3D_Baricenter(float*Triangle,float*Result);
;************************
;Given a triangle Triangle described by and array of 3 3D points (3 floats per point)
;this algorithm calculates its baricenter and returns and array of a 3D point in Result
;************************
Triangle_3D_Baricenter:
    enter 0,0    
    movups XMM0,[arg1]
    add arg1,(4*3) ;<- It jumps three times the size of a float (4 bytes)
    movups XMM1,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float because of memory boundaries
    movups XMM2,[arg1] 
    movss XMM3,fc_3f_mem
    ;xmm0 [Bx][Az][Ay][Ax]
    ;xmm1 [Cx][Bz][By][Bx]
    ;xmm2 [Cz][Cy][Cx][Bz]  ;<- It needs some alingment
    ;xmm3 [??][??][??][3.0f];<- It needs to be in all sections
    psrldq XMM2,4    
    pshufd xmm3,xmm3,0
    ;xmm0 [Bx][Az][Ay][Ax]
    ;xmm1 [Cx][Bz][By][Bx]
    ;xmm2 [??][Cz][Cy][Cx] :<- it's aligned now
    ;xmm3 [?][3.0f][3.0f][3.0f]
    addps xmm0,xmm1
    addps xmm1,xmm2
    divps xmm0,xmm3
    movhlps xmm1,xmm0
    ;xmm0 [??][Rz][Ry][Rx]
    ;xmm1 [??][??][??][Rz]
    movsd [arg2],xmm0
    add arg2,(4*2)
    movss [arg2],xmm1
    leave
    ret

global Triangle_2D_Baricenter; Triangle_2D_Baricenter(float*Triangle,float*Result);
;************************
;Given a triangle Triangle described by and array of 3 2D points (2 floats per point)
;this algorithm calculates its baricenter and returns and array of a 2D point in Result
;************************
Triangle_2D_Baricenter:
    enter 0,0    
    movsd XMM0,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float (4 bytes)
    movsd XMM1,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float because of memory boundaries
    movsd XMM2,[arg1] 
    movss XMM3,fc_3f_mem
    ;xmm0 [??][??][Ay][Ax]
    ;xmm1 [??][??][By][Bx]
    ;xmm2 [??][??][Cx][Bz]  
    ;xmm3 [??][??][??][3.0f];<- It needs to be in all sections
    movsldup xmm3,xmm3
    ;xmm3 [?][?][3.0f][3.0f]
    addps xmm0,xmm1
    addps xmm1,xmm2
    divps xmm0,xmm3
    ;xmm0 [??][??][Ry][Rx]
    movsd [arg2],xmm0
    leave
    ret


global Check_Point_in_Triangle_2D; char Check_2D_Point_in_Triangle(float * 2D_Triangle, float * 2D_Point);
;***************
;Given a 2D Triangle and a 2D Point,
;this algorithm returns 1 if the point is inside the Triangle boundaries
;else, this algoritm returns 0 
;***************
Check_Point_in_Triangle_2D:
    enter 0,0
    xor RAX,RAX
   
    movsd xmm0,[arg2]		    ;(P.x,P.y)
    movsd xmm1,[arg1]		    ;(A.x,A.y)
    movsd xmm2,[arg1+(4*3)]	    ;(B.x,B.y)
    movsd xmm3,[arg1+(4*3)+(4*3)]   ;(C.x,C.y)

    movsd xmm4,xmm0
    subss xmm4,xmm1 ;xmm4=P-A
    movsd xmm5,xmm2
    subss xmm5,xmm1 ;xmm5=B-A

    DotProductXMMV2 xmm4,xmm5,xmm6,xmm7
    ;xmm6 = PAB
 
    movsd xmm4,xmm0
    subss xmm4,xmm2 ;xmm4=P-B
    movsd xmm5,xmm3
    subss xmm5,xmm2 ;xmm5=C-B

    DotProductXMMV2 xmm4,xmm5,xmm2,xmm7
    ;xmm2 = PBC
    
    MOVMSKPS arg1,xmm6
    MOVMSKPS arg2,xmm2

    and arg1,1 ; 1 = Negative, 0 = Positive
    and arg2,1 ; 1 = Negative, 0 = Positive
    cmp arg1,arg2 
    jne final

    
    movsd xmm4,xmm0
    subss xmm4,xmm3 ;xmm4=P-C
    movsd xmm5,xmm1
    subss xmm5,xmm3 ;xmm5=A-C

    DotProductXMMV2 xmm4,xmm5,xmm2,xmm7
    ;xmm2 = PCA

    MOVMSKPS arg2,xmm2

    and arg1,1 ; 1 = Negative, 0 = Positive
    and arg2,1 ; 1 = Negative, 0 = Positive
    cmp arg1,arg2 
    jne final

    neg RAX
    shr RAX,63

final:
    leave
    ret


%endif
