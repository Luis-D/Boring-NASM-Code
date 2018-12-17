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

; December 2th, 2018:	First version
; December 3th, 2018:	New function added.
; December 6th, 2018:	Check_Segment_vs_Segment_2D fixed
; December 12th, 2018:	"Baricenter" -> "Barycenter"
; December 14th, 2018:	Offset capability ti Check_Point_in_Triangle_2D added

; Luis Delgado.

; Collection of mathematical functions, very useful for geometry.


%ifndef _NASMGEOMETRY_ASM_
%define _NASMGEOMETRY_ASM_

%include "NasmMath64.asm" ; Compile with -i 

;*****************************
;MACROS
;*****************************

%ifidn __OUTPUT_FORMAT__, elf64 
%elifidn __OUTPUT_FORMAT__, win64
%endif

%ifidn __OUTPUT_FORMAT__, win64 
%endif

args_reset ;<--Sets arguments definitions to normal, as it's definitions can change.

%macro CROSSPRODUCTV2 4
;%1 and %2 Registers to operate with
;%3 Register where to store the result
;%4 Temporal register
;v = %1; w = %2

pshufd %4,%2,00000001b
movsd %3,%1
mulps %3,%4
movshdup %4, %3
subss %3,%4

%endmacro

;********************************************************************
; .data
;********************************************************************

fc_3f_mem: dd 01000000010000000000000000000000b ; 3.0f;
fc_1f_mem: dd 0x3f800000;1.0f;

;********************************
; CODE
;********************************

section .text

global Triangle_3D_Barycenter; Triangle_3D_Barycenter(float*Triangle,float*Result);
;************************
;Given a triangle Triangle described by and array of 3 3D points (3 floats per point)
;this algorithm calculates its barycenter and returns and array of a 3D point in Result
;************************
Triangle_3D_Barycenter:
    enter 0,0    
    movups XMM0,[arg1]
    add arg1,(4*3) ;<- It jumps three times the size of a float (4 bytes)
    movups XMM1,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float because of memory boundaries
    movups XMM2,[arg1] 
    movss XMM3, [fc_3f_mem]
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

global Triangle_2D_Barycenter; Triangle_2D_Barycenter(float*Triangle,float*Result);
;************************
;Given a triangle Triangle described by and array of 3 2D points (2 floats per point)
;this algorithm calculates its barycenter and returns and array of a 2D point in Result
;************************
Triangle_2D_Barycenter:
    enter 0,0    
    movsd XMM0,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float (4 bytes)
    movsd XMM1,[arg1]
    add arg1,(4*2) ;<- It jumps two times the size of a float because of memory boundaries
    movsd XMM2,[arg1] 
    movss XMM3, [fc_3f_mem]
    ;xmm0 [??][??][Ay][Ax]
    ;xmm1 [??][??][By][Bx]
    ;xmm2 [??][??][Cx][Bz]  
    ;xmm3 [??][??][??][3.0f];<- It needs to be in all sections
    movsldup xmm3,xmm3
    ;xmm3 [?][?][3.0f][3.0f]
    addps xmm0,xmm1
    addps xmm0,xmm2
    divps xmm0,xmm3
    ;xmm0 [??][??][Ry][Rx]
    movsd [arg2],xmm0
    leave
    ret

global Check_Point_in_Segment_2D; char Check_Point_in_Segment_2D(float * Segment, float * 2D_Point)
Check_Point_in_Segment_2D:
    enter 0,0
    xor RAX,RAX    
    
    movsd xmm0,[arg2]		    ;(P.x,P.y)
    movsd xmm1,[arg1]		    ;(A.x,A.y)
    movsd xmm2,[arg1+(4*2)]	    ;(B.x,B.y)
    pxor xmm3,xmm3

    movsd xmm4,xmm0
    subps xmm4,xmm1 ;xmm4=P-A
    movsd xmm5,xmm2
    subps xmm5,xmm1 ;xmm5=B-A


    CROSSPRODUCTV2 xmm4,xmm5,xmm0,xmm7
    ;xmm0 = AP x AB

    DotProductXMMV2 xmm5,xmm4,xmm1,xmm7
    ;xmm1 = AB . AP
    
    DotProductXMMV2 xmm5,xmm5,xmm2,xmm7
    ;xmm2 = AB . AB
    
    cmpss xmm0,xmm3,0 ;xmm0= if xmm0 = 0
    cmpss xmm3,xmm1,2 ;xmm3= if 0 <= xmm1    
    cmpss xmm1,xmm2,2 ;xmm1= if xmm1 <= xmm2
	
    andps xmm1,xmm3
    andps xmm0,xmm1

    sub rsp,8
    movss [rsp],xmm0
    mov eax,[rsp]
    add rsp,8

    cmp eax,0xFFFFFFFF
    je __set1 

    xor rax,rax
    jmp __final

__set1:
    mov rax,1

__final:
    leave
    ret

global Check_Point_in_Triangle_2D; char Check_Point_in_Triangle_2D(float * 2D_Triangle, float * 2D_Point,int bytes_offset);
;***************
;Given a 2D Triangle and a 2D Point,
;this algorithm returns 1 if the point is inside the Triangle boundaries
;else, this algoritm returns 0
;The offset can be used if the triangle vertices elements are interleaved with something else 
;***************
Check_Point_in_Triangle_2D:
    enter 0,0
    xor RAX,RAX
   
    movsd xmm0,[arg2]		    ;(P.x,P.y)
    movsd xmm1,[arg1]		    ;(A.x,A.y)
    add arg1,arg3
    movsd xmm2,[arg1+(4*2)]	    ;(B.x,B.y)
    add arg1,arg3
    movsd xmm3,[arg1+(4*2)+(4*2)]   ;(C.x,C.y)

    movsd xmm4,xmm0
    subps xmm4,xmm1 ;xmm4=P-A
    movsd xmm5,xmm2
    subps xmm5,xmm1 ;xmm5=B-A

    CROSSPRODUCTV2 xmm5,xmm4,xmm6,xmm7
    ;xmm6 = PAB

 
    movsd xmm4,xmm0
    subps xmm4,xmm2 ;xmm4=P-B
    movsd xmm5,xmm3
    subps xmm5,xmm2 ;xmm5=C-B

    CROSSPRODUCTV2 xmm5,xmm4,xmm2,xmm7
    ;xmm2 = PBC
    
    movsd xmm4,xmm0
    subps xmm4,xmm3 ;xmm4=P-C
    movsd xmm5,xmm1
    subps xmm5,xmm3 ;xmm5=A-C

    CROSSPRODUCTV2 xmm5,xmm4,xmm3,xmm7
    ;xmm3 = PCA

    xor arg1,arg1
    xor arg2,arg2
    xor arg3,arg3
    
    pxor xmm0,xmm0
    MOVMSKPS arg3,xmm3
    MOVMSKPS arg1,xmm6
    MOVMSKPS arg2,xmm2

    and arg1,1 ; 1 = Negative, 0 = Positive
    and arg2,1 ; 1 = Negative, 0 = Positive
    and arg3,1 ; 1 = Negative, 0 = Positive
    cmp arg1,arg2 
    jne final
    cmp arg1,arg3
    jne final

    mov rax,1 
final:
    leave
    ret

global Check_Segment_vs_Segment_2D
;char Check_Segment_vs_Segment_2D(float * Seg_A, float * Seg_B, float * Time_Return);
;Seg_A = Q -> S
;Seg_B = P -> R
Check_Segment_vs_Segment_2D:
    enter 0,0
    
    xor RAX,RAX

    movsd xmm0,[arg2]	;xmm0=Q
    movsd xmm2,[arg1]	;xmm2=P
    add arg2,8    
    movsd xmm1,[arg2]	;xmm1=Q+S
    subps xmm1,xmm0     ;xmm1=Q
    add arg1,8
    subps xmm0,xmm2	;xmm0 = Q-P
    movsd xmm3,[arg1]   ;xmm3 = P+R
    subps xmm3,xmm2     ;xmm3 = P

    CROSSPRODUCTV2 xmm0,xmm1,xmm2,xmm4
    ;xmm2 = (Q-P) x S
    
    CROSSPRODUCTV2 xmm3,xmm1,xmm4,xmm5
    ;xmm4 = (R x S)

    CROSSPRODUCTV2 xmm0,xmm3,xmm1,xmm5
    ;xmm1 = (Q-P) x R
  
 
%if 1
 
    sub rsp,8
    movss [rsp],xmm4
    mov eax,[rsp]

    divss xmm2,xmm4;xmm2 = t = (Q-P) x S / (R x S)
    

    divss xmm1,xmm4;xmm1 = u = (Q-P) x R / (R x S)
   
    movss xmm0,xmm2;<- save to return

    movlhps xmm1,xmm2
    ;xmm1 = [?][t][?][u]
    pshufd xmm1,xmm1,10_0_10_00b

    cmp eax,0
    je _final 
    mov rax,0
    
    sub rsp,8

    pxor xmm2,xmm2
    movss xmm3,[fc_1f_mem]
    pshufd xmm3,xmm3,0
   
    ;movups [arg3],xmm1

    cmpps xmm2,xmm1,2 ;xmm2= if 0 <= xmm1    
    cmpps xmm1,xmm3,2 ;xmm1= if xmm1 <= 1.f

    ;movups [arg3],xmm2

    movlhps xmm2,xmm1
    ;xmm2 [t<1.f][u<1.f][0<t][0<u]

    ;movups [arg3],xmm2    
    
    movups [rsp],xmm2
    mov arg1,[rsp]
    add rsp,8
    mov arg2,[rsp]
    add rsp,8
    
    mov arg4,0xFFFFFFFFFFFFFFFF

    cmp arg1,arg4
    jne _final
    cmp arg2,arg4
    jne _final

    mov eax,1
    cmp arg3,0
    je _final
    movss [arg3],xmm0
%endif
   
_final:
    leave
    ret

%endif
