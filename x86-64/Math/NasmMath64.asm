; To be used with NASM-Compatible Assemblers

; System V AMD64 ABI Convention

; Function parameters are passed this way:
; Interger values: RDI, RSI, RDX, RCX, R8, R9, R10, and R11
; Float Point values: XMM0, XMM1, XMM2, XMM3 

; Function return value is returned this way:
; Integer value: RAX:RDX 
; Float Point value: XMM0

; SSE, SSE2, SSE3

; Luis Delgado. November 26, 2018 (Date of last edition).

; Collection of mathematical functions, very useful for algebra.



;*****************************
;MACROS
;*****************************

%macro TRANS44 0
    movaps  xmm4,  xmm0
    movaps  xmm6,  xmm2

    punpckldq xmm0,xmm1
    punpckldq xmm2,xmm3
    punpckhdq xmm4,xmm1
    punpckhdq xmm6,xmm3

    movaps  xmm1,xmm0
    movaps  xmm5,xmm4

    punpcklqdq xmm0,xmm2
    punpckhqdq xmm1,xmm2
    punpcklqdq xmm4,xmm6
    punpckhqdq xmm5,xmm6
%endmacro

%macro CROSSPRODUCTMACRO 6
;%1 First register to use
;%2 Second register to use
;%3 Register to store result
;%4, %5, %6 temporal registers
       movups %3,%1
        ;%3 [?][Az][Ay][Ax]
        movups %4,%2
        ;%4 [?][Bz][By][Bx]
        pshufd %5,%3,11010010b
        pshufd %6,%4,11001001b
        pshufd %3,%3,11001001b
        pshufd %4,%4,11010010b
        ;%3 [?][Bx][Bz][By]
        ;%4 [?][By][Bx][Bz]
        ;%5 [?][Ay][Ax][Az]
        ;%6 [?][Bx][Bz][By]
        mulps %3,%4
        mulps %5,%6
        subps %3,%5
        ;%3 [?][Rz][Ry][Rx]
%endmacro

%macro NORMALIZEVEC3MACRO 4 
;%1 Register with the vector about to normalize
;%2, %3 y %4 temporal registers

    movaps %3,%1
    ;%3 [][z][y][x]
    mulps  %3,%1
    ;%3 [][z*z][y*y][x*x]	
	movhlps %2,%3
	;%2 [][][][z*z]
	pshufd %4,%3,1
	;%4 [][][][y*y]
	addss %3,%4
	addss %3,%2
	;%3 [][][][(x*x)+(y*y)+(z*z)]
	sqrtss %3,%3
	;%3 [][][][sqrt((x*x)+(y*y)+(z*z))]
    pshufd %3,%3,0
    divps %1,%3
%endmacro 

%macro MULVEC4VEC4 3
        movups xmm2,[%1+%3]
        movaps xmm7,xmm2

        mulps  xmm2,xmm0
        movshdup    xmm3, xmm2
        addps       xmm2, xmm3
        movaps xmm6,xmm7
        movhlps     xmm3, xmm2
        addss       xmm2, xmm3
        movss  [%2+%3], xmm2;//<--- Important

        mulps  xmm6,xmm1
        movshdup    xmm3, xmm6
        addps       xmm6, xmm3
        movaps xmm2,xmm7
        movhlps     xmm3, xmm6
        addss       xmm6, xmm3
        movss  [%2+4+%3], xmm6;//<--- Important

        mulps  xmm2,xmm4
        movshdup    xmm3, xmm2
        addps       xmm2, xmm3
        movaps xmm6,xmm7
        movhlps     xmm3, xmm2
        addss       xmm2, xmm3
        movss  [%2+8+%3], xmm2;<--- Important

        mulps  xmm6,xmm5
        movshdup    xmm3, xmm6
        addps       xmm6, xmm3
        movhlps     xmm3, xmm6
        addss       xmm6, xmm3
        movss  [%2+8+4+%3], xmm6;<--- Important

%endmacro

%macro DotProductXMM 4
;%1 and %2 are registers to proccess
;%3 is the result
;%4 is a temporal register
	movaps %3, %1
	mulps  %3, %2
	movshdup %4,%3
	addps    %3,%4
	movhlps  %4,%3
	addss    %3,%4
%endmacro

%macro M4x4MULMACRO 2 ;When used, registers should be 0'd
    DotProductXMM xmm3,%1,xmm8,xmm9
    pshufd xmm10,xmm8,0
    DotProductXMM xmm2,%1,xmm8,xmm9
    movss xmm10,xmm8
    DotProductXMM xmm1,%1,xmm8,xmm9
    pshufd %2,xmm8,0
    DotProductXMM xmm0,%1,xmm8,xmm9
    movss %2,xmm8
    movlhps %2,xmm10
%endmacro


;********************************************************************
; .data
;********************************************************************

section .data

  ;***** Constants *****;
    fc_360f:                   equ 0x43b40000                         ;32-bits 360.f
    fc_2f:                     equ 01000000000000000000000000000000b  ;32-bits 2.f
    fc_m_1f:                   equ 0xbf800000                         ;32-bits -1.f
    fc_1f:                     equ 0x3f800000                         ;32-bits +1.f
    SignChange32bits:          equ 10000000000000000000000000000000b  ;It can change the sign if XOR'd
    fc_180f:                   equ 0x43340000                         ;32-bits 180.f
    fc_PIdiv180f:              equ 0x3c8efa35                         ;32-bits (PI/180.f)
    fc_180fdivPI:              equ 0x42652ee1                         ;32-bits (180.f/PI)

;***** Variables *****;
    fc_PIdiv180f_mem:           dd 0x3c8efa35                     ;32-bits (PI/180.f)




;********************************
; CODE
;********************************

section .text


global V2Rotate_FPU; 
;void V2Rotate_FPU(float * Vec2_A, float * Angle_Degrees, float * Vec2_Result);
;****************************************************************
It rotates 2D point A around (0,0) by and angle given in degrees.
;****************************************************************
V2Rotate_FPU:
    enter 0,0

    sub rsp,8

    fld dword [fc_PIdiv180f_mem]
        pxor xmm4,xmm4
        pxor xmm3,xmm3
        
    fmul dword [rsi]
    pcmpeqd xmm4,xmm4
        ;xmm4[0] = [11111111111111111111111111111111]
        
    fld st0
    fcos 
    fstp dword [rsp]
       pslld xmm4,31
            ;xmm4[0] = [10000000000000000000000000000000]
          movss xmm3,xmm4
    fsin
    fstp dword [rsp+4]

    movsd xmm0,[rsp]
    ;xmm0 [][][sin][cos]
    pshufd xmm1,xmm0,11_10_00_01b
    ;xmm1 [][][cos][sin]
    movsd xmm2,[rdi]
    movsldup xmm5,xmm2
    ;xmm5 [][][x][x]
    movshdup xmm6,xmm2
    ;xmm6 [][][y][y]
    pxor xmm6,xmm3
    ;xmm6 [][][y][-y]

    mulps xmm0,xmm5
    mulps xmm1,xmm6
    addps xmm0,xmm1

    movsd [rdx],xmm0

    add rsp,8
    leave 
    ret

global V2ScalarMUL;V2ScalarMUL(float * Vec2_A, float Scalar_B, float * Vec2_Result);
;**********************************************
;Given A (2D Vectors) and B (an Scalar value),
;2D Vector Result = (Ax * B , Ay * B);
;**********************************************
V2ScalarMUL:
    enter 0,0
    movsd xmm1, [rdi]
    movsldup xmm0,xmm0
    mulps xmm0,xmm1
    movsd [rsi],xmm0
    leave
    ret 


global V2V2ADD; V2V2ADD(float * Vec2_A, float * Vec2_B, float * Vec2_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax + Bx , Ay + By);
;**********************************************
V2V2ADD:
    enter 0,0
    movsd xmm0, [rdi]
    movsd xmm1, [rsi]
    addps xmm0,xmm1
    movsd [rdx],xmm0
    leave
    ret

global V2V2SUB; V2V2SUB(float * Vec2_A, float * Vec2_B, float * Vec2_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax - Bx , Ay - By);
;**********************************************
V2V2SUB:
    enter 0,0
    movsd xmm0, [rdi]
    movsd xmm1, [rsi]
    subps xmm0,xmm1
    movsd [rdx],xmm0
    leave
    ret


global V3V3ADD; V3V3ADD(float* A, float* B, float* Resultado);
;*********************************************************
;Given A and B (both 3D vectors),
;3D Vector Result = (Ax + Bx, Ay + By, Az + Bz, Aw + Bw);
;*********************************************************
V3V3ADD:
    enter 0,0
	movups xmm0,[rdi]
        movups xmm1,[rsi]
        addps xmm0,xmm1
	movhlps xmm1,xmm0
        movsd [rdx],xmm0
	movss [rdx+8],xmm1
    leave
    ret

global V3V3SUB;  void V3V3SUB(float * Vec3_A, float * Vec3_B,float * Vec3_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax - Bx , Ay - By);
;**********************************************
V3V3SUB:
    enter 0,0
    movsd xmm0, [rdi]
    movsd xmm1, [rsi]
    subps xmm0,xmm1

    movss xmm1,[rdi+8]
    movss xmm2,[rsi+8]
    subss xmm1,xmm2

    movsd [rdx],xmm0
    movss [rdx+8],xmm1


    leave
    ret


global V4V4ADD; void V4V4ADD(float * A, float *B, float * Result);
;*********************************************************
;Given A and B (both 4D vectors),
;4D Vector Result = (Ax + Bx, Ay + By, Az + Bz, Aw + Bw);
;*********************************************************
V4V4ADD:
    enter 0,0
        movups xmm0,[rdi]
        movups xmm1,[rsi]
        addps xmm0,xmm1
        movups [rdx],xmm0
    leave
    ret
    
global V4V4SUB; void V4V4ADD(float * A, float *B, float * Result);
;*********************************************************
;Given A and B (both 4D vectors),
;4D Vector Result = (Ax - Bx, Ay - By, Az - Bz, Aw - Bw);
;*********************************************************
V4V4ADD:
    enter 0,0
        movups xmm0,[rdi]
        movups xmm1,[rsi]
        subps xmm0,xmm1
        movups [rdx],xmm0
    leave
    ret


global V2CalculatePerpendicular
; V2CalculatePerpendicular(float * Vec2_A,float * Vec2_Result,float ClockWise_Multiplier)
;**********************************************
;Given a 2D Vector A,
;This function calculates a perpendicular vector. Direction depends on the third parameter:
;ClockWise_Multiplier should be 1.f or -1.f
;**********************************************
V2CalculatePerpendicular:
    enter 0,0

    movsd xmm1, [rdi]
    movsldup xmm0,xmm0
    
    pxor xmm4,xmm4
    pcmpeqd xmm4,xmm4
    ;xmm4[0] = [11111111111111111111111111111111]
    pxor xmm3,xmm3
    pslld xmm4,31
    ;xmm4[0] = [10000000000000000000000000000000]

    movss xmm3,xmm4
  
    pxor xmm1,xmm3
    pshufd xmm1,xmm1,11_10_00_01b
    mulps xmm1,xmm0

    movsd [rsi],xmm1

    leave 
    ret



global V2V2Dot; float DotProduct = V2V2Dot (float * Vec2_A, float * Vec2_B);
;**********************************************
;Given A and B 2D Vectors,
;it returns (A . B)
;**********************************************
V2V2Dot:
    enter 0,0
    movsd xmm2,[rdi]
    pxor xmm0,xmm0
    movsd xmm1,[rsi]
    mulps xmm2,xmm1
    movshdup xmm1,xmm2
    addss xmm2,xmm1

    movss xmm0,xmm2 ;Return is stored in XMM0
    
    leave
    ret
    

global M4x4MUL ;(float * A, float *B, float * Result);
;**********************************************
;Given A and B (both 4x4 Matrices),
;4x4 Matrix Result = A * B;
;**********************************************
M4x4MUL:
    enter 0,0
        movups  xmm0, [rdi]
        movups  xmm1, [rdi+16]
        movups  xmm2, [rdi+32]
        movups  xmm3, [rdi+16+32]
    TRANS44; Matrix A (rows) in 0,1,4 and 5
	  movaps xmm2,xmm4
	  movaps xmm3,xmm5
    ;Matriz A (rows) in 0,1,2,3

        movups  xmm4, [rsi]
        movups  xmm5, [rsi+16]
        movups  xmm6, [rsi+32]
        movups  xmm7, [rsi+16+32]
    ;Matriz B (Columns) in 4,5,6,7

    M4x4MULMACRO xmm4,xmm12
    M4x4MULMACRO xmm5,xmm13
    M4x4MULMACRO xmm6,xmm14
    M4x4MULMACRO xmm7,xmm15

    movups [rdx],xmm12
    movups [rdx+16],xmm13
    movups [rdx+32],xmm14
    movups [rdx+16+32],xmm15

    leave
    ret


global NormalizeDoubleVec2; NormalizeVec2(double * A, double * Result)
;******************************************************
;Given a 2D Vector A whose elements are 64-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeDoubleVec2:
    enter 0,0
        movupd xmm0,[rdi]
        movapd xmm1,xmm0
        mulpd xmm1,xmm0
        movhlps xmm2,xmm1
        addsd xmm1,xmm2
        sqrtsd xmm1,xmm1
        punpcklqdq xmm1,xmm1
        divpd xmm0,xmm1
        movupd [rsi],xmm0
    leave
    ret

global NormalizeVec2; NormalizeVec2(Float * A, float * Result)
;******************************************************
;Given a 2D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec2:
    enter 0,0
        movq xmm0,[rdi]
        movq xmm1,xmm0
        mulps xmm1,xmm0
        movshdup xmm2,xmm1
        addss xmm1,xmm2
        sqrtss xmm1,xmm1
        movsldup xmm1,xmm1
        divps xmm0,xmm1
        movq [rsi],xmm0
    leave
    ret


global NormalizeVec3; NormalizeVec3(Float * A, float * Result)
;******************************************************
;Given a 3D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec3:
    enter 0,0
        movups xmm0,[rdi]
        ;xmm0 [][z][y][x]
        movaps xmm1,xmm0
        ;xmm1 [][z][y][x]
        mulps  xmm1,xmm0
        ;xmm1 [][z*z][y*y][x*x]
        movaps xmm3,xmm0
        ;xmm3 [][z][y][x]
        movhlps xmm0,xmm1
        ;xmm0 [][][][z*z]
        pshufd xmm2,xmm1,1
        ;xmm2 [][][][y*y]
        addss xmm1,xmm2
        addss xmm1,xmm0
        ;xmm1 [][][][(x*x)+(y*y)+(z*z)]
        sqrtss xmm1,xmm1
        ;xmm1 [][][][sqrt((x*x)+(y*y)+(z*z))]
        movlps [rsi],xmm3
        movss [rsi+8],xmm1
    leave
    ret

global NormalizeVec4; NormalizeVec3(Float * A, float * Result)
;******************************************************
;Given a 4D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec4:
    enter 0,0
        movups xmm0,[rdi]
        ;xmm0 [w][z][y][x]
        movaps xmm1,xmm0
        ;xmm1 [w][z][y][x]
        mulps  xmm1,xmm0
        ;xmm1 [w*w][z*z][y*y][x*x]
        movshdup xmm2,xmm1
        addps    xmm1,xmm2
        movhlps  xmm2,xmm1
        addss    xmm1,xmm2
        sqrtss   xmm1,xmm1
        pshufd   xmm1,xmm1,0
        divps    xmm0,xmm1
        movups [rsi],xmm0
    leave
    ret

global CrossProductVec3; CrossProductVec3(float * A, float * B, float * Result)
;******************************************************
;Given A and B (both 3D Vectors)
;3D Vector Result = (A x B)
;******************************************************
CrossProductVec3:
    enter 0,0
        movups xmm0,[rdi]
        ;xmm0 [?][Az][Ay][Ax]
        movups xmm1,[rsi]
        ;xmm1 [?][Bz][By][Bx]
        pshufd xmm2,xmm0,11010010b
        pshufd xmm3,xmm1,11001001b
        pshufd xmm0,xmm0,11001001b
        pshufd xmm1,xmm1,11010010b
        ;xmm0 [?][Bx][Bz][By]
        ;xmm1 [?][By][Bx][Bz]
        ;xmm2 [?][Ay][Ax][Az]
        ;xmm3 [?][Bx][Bz][By]
        mulps xmm0,xmm1
        mulps xmm2,xmm3
        subps xmm0,xmm2
        ;xmm0 [?][Rz][Ry][Rx]
        movhlps xmm1,xmm0
        movlps [rdx],xmm0
        movss [rdx+8],xmm1
    leave
    ret

global M4x4V4MUL;M4x4V4MUL(float * MatrixA, float *VectorB, float * Result);
;******************************************************
; Given a 4x4 Matrix MatrixA and a 4D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB;
;******************************************************
M4x4V4MUL: 
    enter 0,0
        movups  xmm0, [rdi]
        movups  xmm1, [rdi+16]
        movups  xmm2, [rdi+32]
        movups  xmm3, [rdi+16+32]
    TRANS44
    movups xmm7,[rsi]
	movaps xmm2,xmm4
	movaps xmm3,xmm5
    M4x4MULMACRO xmm7,xmm15
	movups [rdx],xmm15
    leave
    ret

; 1.0f = 00111111100000000000000000000000

global M4x4V4_PseudoV3_W1_MUL
; M4x4V4_PseudoV3_W1_MUL(float * MatrixA, float *VectorB, float * Result);
;*******************************************************************************************
; Given a 4x4 Matrix MatrixA and a 3D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB, BUT:
; - In order for the operation to be possible, a 4th element (1.f) is added to VectorB
;*******************************************************************************************
M4x4V4_PseudoV3_W1_MUL: 
    enter 0,0
        movups  xmm0, [rdi]
        movups  xmm1, [rdi+16]
        movups  xmm2, [rdi+32]
        movups  xmm3, [rdi+16+32]
    TRANS44
    movups xmm7,[rsi]
	movaps xmm2,xmm4
	movaps xmm3,xmm5

    pxor xmm4,xmm4
    pcmpeqd xmm4,xmm4
    ;xmm4[0] = [11111111111111111111111111111111]
    psrld xmm4,25
    ;xmm4[0] = [00000000000000000000000001111111]
    pslld xmm4,23
    ;xmm4[0] = [00111111100000000000000000000000] = 1.f
    ;xmm4 = [1.f][1.f][1.f][1.f]


    movhlps xmm5,xmm7
    ;xmm5 = [?][?][W][Z]
    movss xmm4,xmm5
    ;xmm4 = [1.f][1.f][1.f][Z]
    movlhps xmm7,xmm4
    ;xmm7 = [1.f][Z][Y][X]
    movlhps xmm5,xmm5
    ;xmm5 = [W][Z][?][?]


    M4x4MULMACRO xmm7,xmm15

    movhlps xmm5,xmm15
    ;xmm5 = [W][Z][Rw][Rz]
    pshufd xmm5,xmm5,11_10_11_00b
    ;xmm5 = [W][Z][W][Rz]
    movlhps xmm15,xmm5
    ;xmm15 = [W][Rz][Ry][Rx]

	movups [rdx],xmm15
    leave
    ret

global M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL
;M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL(float * MatrixA, float *VectorB, float * Resultado);
;****************************************************************************************
; Given a 4x4 Matrix MatrixA and a 3D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB, BUT:
; - In order for the operation to be possible, a 4th element (1.f) is added to VectorB
; - The last row and column of the matrix are temporarily replaced both with (0,0,0,1.f)
; - This function calculates only the affine transformation.
;****************************************************************************************
M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL: 
     enter 0,0
        movups  xmm0, [rdi]
        movups  xmm1, [rdi+16]
        movups  xmm2, [rdi+32]

    pxor xmm14,xmm14
    pcmpeqd xmm14,xmm14
    ;xmm14[0] = [11111111111111111111111111111111]
    psrld xmm14,25
    ;xmm14[0] = [00000000000000000000000001111111]
    pslld xmm14,23
    ;xmm14[0] = [00111111100000000000000000000000] = 1.f
    ;xmm14 = [1.f][1.f][1.f][1.f]

    pxor xmm3,xmm3
    movss xmm3,xmm14
    pshufd xmm3,xmm3, 00_10_01_11b

    TRANS44
    movups xmm7,[rsi]
	  movaps xmm2,xmm4

    pxor xmm3,xmm3
    movss xmm3,xmm14

    movhlps xmm5,xmm7
    ;xmm5 = [?][?][W][Z]
    movss xmm14,xmm5
    ;xmm14 = [1.f][1.f][1.f][Z]
    movlhps xmm7,xmm14
    ;xmm7 = [1.f][Z][Y][X]
    movlhps xmm5,xmm5
    ;xmm5 = [W][Z][?][?]


    M4x4MULMACRO xmm7,xmm15

    movhlps xmm5,xmm15
    ;xmm5 = [W][Z][Rw][Rz]
    pshufd xmm5,xmm5,11_10_11_00b
    ;xmm5 = [W][Z][W][Rz]
    movlhps xmm15,xmm5
    ;xmm15 = [W][Rz][Ry][Rx]

	  movups [rdx],xmm15
    leave
    ret

global QuaternionMUL; (float * A, float * B, float * Result);
;**********************************
; Given A and B (both Quaternions),
; Quaternion Result = A * B;
;**********************************
QuaternionMUL:
    enter 0,0
    mov rcx, SignChange32bits
    push rcx
        movss xmm5,[rsp]


        movups xmm7,[rdi]
        ;xmm7 = [Aw][Az][Ay][Ax]

        movups xmm6,[rsi]
        ;xmm7 = [Bw][Bz][By][Bx]

        pshufd xmm4,xmm6,00_01_10_11b
        ;xmm4 = [Bx][By][Bz][Bw]

        movaps xmm0,xmm4
        mulps xmm0,xmm7

        pshufd xmm4,xmm6,01_00_11_10b
        ;xmm4 = [By][Bx][Bw][Bz]

        movaps xmm1,xmm4
        mulps xmm1,xmm7

        pshufd xmm4,xmm6,10_11_00_01b
        ;xmm4 = [Bz][Bw][Bx][By]

        movaps xmm2,xmm4
        mulps xmm2,xmm7

        movaps xmm3,xmm6
        mulps xmm3,xmm7

        pshufd xmm4,xmm5,10_11_00_01b
        pxor xmm0,xmm4

        pshufd xmm4,xmm5,11_00_11_01b
        pxor xmm1,xmm4

        pshufd xmm4,xmm5,11_11_11_00b
        pxor xmm2,xmm4

        pshufd xmm4,xmm5,11_00_00_00b
        pxor xmm3,xmm4

    TRANS44

        addps xmm0,xmm1
        addps xmm0,xmm4
        addps xmm0,xmm5
        movups [rdx],xmm0
    add rsp,8

    leave
    ret

global QuaternionToMatrix4x4; QuaternionToMatrix4x4(float * Quaternion, float * Matrix)
;********************************************
;Given a Quaternion, his function generates a 4x4 Matrix.
;This algorithm is an implementation of a method by Jay Ryness (2008)
;Source: https://sourceforge.net/p/mjbworld/discussion/122133/thread/c59339da/
;********************************************
QuaternionToMatrix4x4: 
    enter 0,0
    
    mov dword[rsi],SignChange32bits
                movss xmm7,[rsi]
                ;xmm7 [0][0][0][0x800000]

    sub rsp,16
                pshufd xmm0,xmm7,00_11_11_11b
                movups xmm3,[rdi]
                movups xmm1,xmm3
                ;xmmm3 [w][z][y][x]
                movaps xmm2,xmm7
                ;xmm7 [0][0][0][0x800000]
                pxor xmm3,xmm0
                pshufd xmm7,xmm7,0h
                movups [rsp],xmm3
                ;xmm7 [0x800000][0x800000][0x800000][0x800000]
                pshufd xmm2,xmm2,11110000b
                ;xmm2 [0][0][0x800000][0x800000]
    mov rdx,rsp
                movups xmm3,xmm1
                ;xmmm3 [w][z][y][x]
        fld dword[rdx]
    add rdx,4
        fld dword[rdx]
                pxor xmm7,xmm3
                ;xmmm7 [-w][-z][-y][-x]
    add rdx,4
        fld dword[rdx]
    add rdx,4
        fld dword[rdx]
        ;Stack: ST0= -w; ST1=z; ST2=y; ST3=x
        
	;xmm3= [w][z][y][x]
        ;xmm7= [-w][-z][-y][-x]
                movups [rsi+16+16+16],xmm7
                pshufd xmm1,xmm3,01111110b
                ;xmm1 [Y][W][W][Z]
                movaps xmm0,xmm1
                ;xmm0 [Y][W][W][Z]
                punpcklqdq xmm1,xmm7
                ;xmm1 [-Y][-X][W][Z]
                movups [rsi+16],xmm1

                pshufd xmm6,xmm7,00100100b
                ;xmm6 [-X][-Z][-Y][-X]
                ;xmm0 [Y][W][W][Z]
                punpckhdq xmm0,xmm6
                ;xmm0 [-X][Y][-Z][W]
                movaps xmm5,xmm0
                ;xmm5 [-X][Y][-Z][W]
                movups [rsi],xmm0

                ;xmm5 [-X][Y][-Z][W]
                pshufd xmm5,xmm5,01_00_11_10b
                ;xmm5 [-Z][W][-X][Y]
                pxor xmm2,xmm5
                ;xmm5 [-Z][W][X][-Y]
                movups [rsi+16+16],xmm2

        fchs
        fstp dword [rsi+16+16+16+12]
        fstp dword [rsi+16+16+16+12-16]
        fstp dword [rsi+16+16+16+12-16-16]
        fstp dword [rsi+16+16+16+12-16-16-16]

    TRANS44
    
    mov rdi,rsi
    xor rdx,rdx
    mov rcx,rsi
    Mmullabelqua:
        MULVEC4VEC4 rdi,rcx,rdx
        add rdx,16
        cmp rdx,64
        jne Mmullabelqua

    add rsp,16
    leave
    ret

global AxisAngleToQuaternion; AxisAngleToQuaternion(float * Axis, float Degree, float * Quaternion)
;*********************************************************************
;Given a 3D Vector describing an Axis and a angle given in degrees,
;This function calculates the respective quaternion.
;*********************************************************************
AxisAngleToQuaternion:
    enter 0,0
	sub rsp,8
	movss [rsp],xmm0
        push fc_180fdivPI
        fld dword [rsp+8]
        fld dword [rsp]
        fdivp
        fld1
    add rsp,16
        fld1

        faddp
        fdivp
        fld st0
            movups xmm0,[rdi]
        fcos
        fxch

        fsin
        fstp dword[rsi]
            movss xmm1,[rsi]
            pshufd xmm1,xmm1,0h
            mulps xmm0,xmm1
            movups [rsi],xmm0
        fstp dword[rsi+4+4+4]
    leave
    ret

global PerspectiveProjectionMatrix4x4
;(float *matrix, float fovyInDegrees, float aspectRatio,float znear, float zfar);
;*********************************************************************
;It's an implementation of gluPerspective
;rdi=&matrix, xmm0=fovyInDegrees,xmm1=aspectRatio,xmm2=znear,xmm3=zfar
;*********************************************************************
PerspectiveProjectionMatrix4x4: 
enter 0,0

    mov rsi, fc_360f
	    pxor xmm12,xmm12
	    movaps xmm11,xmm3
push rsi
	    mov esi, fc_m_1f
	    pxor xmm10,xmm10
	fldpi
sub rsp,8
	fstp dword[rsp]
	    movss xmm12,xmm2
	    pxor  xmm9,xmm9
	    movss xmm13,[rsp]
	    addss xmm12,xmm12
	    subss xmm11,xmm2
	    movss xmm14,[rsp+8]
	    mulss xmm0,xmm13
	    subss xmm10,xmm12
	    divss xmm0,xmm14
	    movss [rsp],xmm0
	    mulss xmm10,xmm3
	    movss xmm9,xmm11
	fld dword [rsp]
	    divss xmm10,xmm11
	    subss xmm9,xmm3
	fptan
	fstp st0
	fstp dword [rsp]
	    subss xmm9,xmm3
	    divss xmm9,xmm11
	    pxor xmm5,xmm5
;XMM11 = temp4 = ZFAR - ZNEAR
;XMM9  = (-ZFAR - ZNEAR)/temp4
;XMM10 = (-temp * ZFAR) / temp4
;XMM12 = temp  =2.0 * ZNEAR

	    pshufd xmm7,xmm10,11_00_11_11b
	    movss xmm0, [rsp]
	    pshufd xmm6,xmm9, 11_00_11_11b
	    mulss xmm0, xmm2
	    mulss xmm1, xmm0
	    addss xmm0,xmm0
;XMM0 = temp3
	    movss xmm5,xmm12
	    divss xmm5,xmm0
	    pshufd xmm5,xmm5,11_11_00_11b
	    addss xmm1,xmm1
;XMM1 = temp2
	    divss xmm12,xmm1

;Resulting matrix in XMM12,XMM5,XMM6,XMM7

	    movups [rdi],xmm12
	    movups [rdi+16],xmm5
	    movups [rdi+16+16],xmm6
	    movups [rdi+16+16+16],xmm7
	    mov    [rdi+16+16+12],esi

add rsp,8
    leave
    ret

global OrthogonalProjectionMatrix4x4;
;    void OrthogonalProjectionMatrix4x4
;    (float *matrix, float Width, float Height, float znear, float zfar);
;*********************************************************
;It's an implementation of gluOrtho
;rdi=&matrix, xmm0=Width,xmm1=Height,xmm2=znear,xmm3=zfar
;*********************************************************
OrthogonalProjectionMatrix4x4: 
    enter 0,0

    mov rsi,fc_2f
    movss xmm4,xmm3
    subss xmm4,xmm2
    push rsi
    addss xmm3,xmm2
    divss xmm3,xmm4
    movss xmm2,[rsp]
    pxor xmm5,xmm5
    pxor xmm6,xmm6
    movss xmm7,xmm2
    mov rsi,fc_1f
    divss xmm2,xmm0
    movss xmm0,xmm7
    divss xmm7,xmm1
    subss xmm6,xmm0
    subss xmm5,xmm3
    divss xmm6,xmm4

;xmm2 = 2/Width
;xmm3 = (zfar+znear)/(zfar-znear)
;xmm4 = zfar-znear
;xmm5 = -((zfar+znear)/(zfar-znear))
;xmm6 = -2
;xmm7 = 2/Height

    movups [rdi],xmm2
    pxor xmm0,xmm0
    pshufd xmm7,xmm7,11_11_00_11b
    divss xmm6,xmm4
    movups [rdi+16],xmm7
    pshufd xmm5,xmm5,11_00_11_11b
    pshufd xmm6,xmm6,11_00_11_11b
    movups [rdi+16+16],xmm6

    add rsp,8

    movups [rdi+16+16+16],xmm5
    mov [rdi+16+16+16+12],rsi

    leave
    ret

global ViewLookAt
; ViewLookAt(float * matrix, float * Vec3From_EYE, float * Vec3To_CENTER, float * Vec3Up);
;*********************************************************
;It's an implementation of glm::LookAt
;*********************************************************
ViewLookAt: 
enter 0,0
    push rax
    xor eax,eax
    mov eax,fc_m_1f
    push rax 

    pxor xmm3,xmm3

    movups xmm9, [rsi] ;EYE
    movups xmm15, [rdx] ;CENTER
    subps xmm15,xmm9 ;xmm15 = f = CENTER - EYE

    movups xmm14, [rcx]
    ;---Normalize f----;
    NORMALIZEVEC3MACRO xmm15,xmm0,xmm1,xmm2
    ;-------------------;
    ;---Normalize up----;
    NORMALIZEVEC3MACRO xmm14,xmm0,xmm1,xmm2
    ;-------------------;

    ;Resumen:
    ;xmm15 = f
    ;xmm14 = up

    movss xmm8, [rsp]

    ;Cross Product s = f x up;
    CROSSPRODUCTMACRO xmm15,xmm14,xmm13,xmm0,xmm1,xmm2
    ;--------------------------;
    ;Normalize s-----;
    NORMALIZEVEC3MACRO xmm13,xmm0,xmm1,xmm2
    ;-----------------;

    ;Resume:
    ;xmm9 = eye
    ;xmm15 = f
    ;xmm14 = up
    ;xmm13 = s

    pshufd xmm8,xmm8,0
    ;xmm8 [-1.f][-1.f][-1.f][-1.f]

    add rsp,8

    ;Cross Product u = s x f;
    CROSSPRODUCTMACRO xmm13,xmm15,xmm14,xmm0,xmm1,xmm2
    ;-------------------------;

    ;Resume:
    ;xmm9 = eye
    ;xmm15 = f
    ;xmm14 = u
    ;xmm13 = s 

    ;calculate -( s . eye )
    DotProductXMM xmm13,xmm9,xmm12,xmm0
    mulss xmm12,xmm8
    ;------------------------------;

    pop rax

    ;calculate -( u . eye )
    DotProductXMM xmm14,xmm9,xmm11,xmm0
    mulss xmm11,xmm8
    ;------------------------------;

    ;calculate ( f . eye )
    DotProductXMM xmm15,xmm9,xmm10,xmm0
    ;------------------------------;  

    ;do f=-f;
    mulps xmm15,xmm8
    ;----------;

    ;Resume:
    ;xmm8 = [-1][-1][-1][-1]
    ;xmm9 = eye
    ;xmm15 = -f
    ;xmm14 = u
    ;xmm13 = s 
    ;xmm12 = -dot (s,eye)
    ;xmm11 = -dot (u,eye)
    ;xmm10 = +dot (f,eye)

    mulps xmm8,xmm8
    ;xmm8 = [1.f][1.f][1.f][1.f]

    movss xmm8,xmm10
    ;xmm8 = [1.f][1.f][1.f][+dot(f,eye)]
    movlhps xmm8,xmm8
    ;xmm8 = [1.f][+dot(f,eye)][1.f][+dot(f,eye)]
    unpcklps xmm12,xmm11
    ;xmm12 = [-dot (u,eye)][-dot (s,eye)][-dot (u,eye)][-dot (s,eye)]
    movsd xmm8,xmm12
    ;xmm8 [1.f][+dot(f,eye)][-dot (u,eye)][-dot (s,eye)]

    movaps xmm0,xmm13
    movaps xmm1,xmm14
    movaps xmm2,xmm15

    TRANS44

    movaps [rdi],xmm0
    add rdi,16
    movaps [rdi],xmm1
    add rdi,16
    movaps [rdi],xmm4
    add rdi,16
    movaps [rdi],xmm8

leave
ret
