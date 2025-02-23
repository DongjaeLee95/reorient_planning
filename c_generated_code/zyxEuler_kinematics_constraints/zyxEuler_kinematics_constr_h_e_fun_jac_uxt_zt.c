/* This file was automatically generated by CasADi 3.6.4.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[33] = {4, 6, 0, 4, 8, 12, 16, 20, 24, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s4[3] = {6, 0, 0};

/* zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt:(i0[4],i1[],i2[],i3[6])->(o0[6],o1[4x6],o2[6x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a4, a5, a6, a7, a8, a9;
  a0=-3.3333333333333337e-01;
  a1=arg[0]? arg[0][1] : 0;
  a2=cos(a1);
  a3=arg[3]? arg[3][0] : 0;
  a4=arg[0]? arg[0][2] : 0;
  a3=(a3-a4);
  a4=(a2*a3);
  a5=sin(a1);
  a6=arg[3]? arg[3][2] : 0;
  a5=(a5*a6);
  a4=(a4-a5);
  a5=(a0*a4);
  a7=-5.7735026918962584e-01;
  a8=sin(a1);
  a9=arg[0]? arg[0][0] : 0;
  a10=sin(a9);
  a11=(a8*a10);
  a12=(a11*a3);
  a13=cos(a9);
  a14=arg[3]? arg[3][1] : 0;
  a15=arg[0]? arg[0][3] : 0;
  a14=(a14-a15);
  a15=(a13*a14);
  a12=(a12+a15);
  a15=cos(a1);
  a16=(a15*a10);
  a16=(a16*a6);
  a12=(a12+a16);
  a16=(a7*a12);
  a5=(a5+a16);
  a16=1.9245008972987529e-01;
  a17=cos(a9);
  a18=(a8*a17);
  a19=(a18*a3);
  a20=sin(a9);
  a21=(a20*a14);
  a19=(a19-a21);
  a21=(a15*a17);
  a21=(a21*a6);
  a19=(a19+a21);
  a21=(a16*a19);
  a5=(a5+a21);
  a21=-7.7363015094695409e-01;
  a22=arg[3]? arg[3][3] : 0;
  a21=(a21*a22);
  a5=(a5+a21);
  a21=-1.3399667277073046e+00;
  a23=arg[3]? arg[3][4] : 0;
  a24=(a21*a23);
  a5=(a5+a24);
  a24=-1.1666726806031675e+00;
  a25=arg[3]? arg[3][5] : 0;
  a26=(a24*a25);
  a5=(a5+a26);
  if (res[0]!=0) res[0][0]=a5;
  a5=(a0*a4);
  a26=5.7735026918962584e-01;
  a27=(a26*a12);
  a5=(a5+a27);
  a27=1.9245008972987523e-01;
  a28=(a27*a19);
  a5=(a5+a28);
  a28=7.7363015094695431e-01;
  a28=(a28*a22);
  a5=(a5+a28);
  a21=(a21*a23);
  a5=(a5+a21);
  a21=1.1666726806031675e+00;
  a28=(a21*a25);
  a5=(a5+a28);
  if (res[0]!=0) res[0][1]=a5;
  a5=6.6666666666666674e-01;
  a28=(a5*a4);
  a29=2.1366252070928492e-17;
  a30=(a29*a12);
  a28=(a28+a30);
  a30=(a27*a19);
  a28=(a28+a30);
  a30=1.5472603018939082e+00;
  a30=(a30*a22);
  a28=(a28+a30);
  a30=-1.2219167078069951e-16;
  a30=(a30*a23);
  a28=(a28+a30);
  a30=-1.1666726806031671e+00;
  a30=(a30*a25);
  a28=(a28+a30);
  if (res[0]!=0) res[0][2]=a28;
  a28=(a0*a4);
  a30=(a7*a12);
  a28=(a28+a30);
  a30=(a16*a19);
  a28=(a28+a30);
  a30=7.7363015094695420e-01;
  a30=(a30*a22);
  a28=(a28+a30);
  a30=1.3399667277073046e+00;
  a31=(a30*a23);
  a28=(a28+a31);
  a21=(a21*a25);
  a28=(a28+a21);
  if (res[0]!=0) res[0][3]=a28;
  a28=(a0*a4);
  a12=(a26*a12);
  a28=(a28+a12);
  a12=(a27*a19);
  a28=(a28+a12);
  a12=-7.7363015094695420e-01;
  a12=(a12*a22);
  a28=(a28+a12);
  a30=(a30*a23);
  a28=(a28+a30);
  a24=(a24*a25);
  a28=(a28+a24);
  if (res[0]!=0) res[0][4]=a28;
  a4=(a5*a4);
  a19=(a27*a19);
  a4=(a4+a19);
  a19=-1.5472603018939084e+00;
  a19=(a19*a22);
  a4=(a4+a19);
  a19=1.7178040122510878e-16;
  a19=(a19*a23);
  a4=(a4+a19);
  a19=1.1666726806031671e+00;
  a19=(a19*a25);
  a4=(a4+a19);
  if (res[0]!=0) res[0][5]=a4;
  a4=cos(a9);
  a19=(a8*a4);
  a19=(a3*a19);
  a25=sin(a9);
  a25=(a14*a25);
  a19=(a19-a25);
  a4=(a15*a4);
  a4=(a6*a4);
  a19=(a19+a4);
  a4=(a7*a19);
  a25=sin(a9);
  a8=(a8*a25);
  a8=(a3*a8);
  a9=cos(a9);
  a14=(a14*a9);
  a8=(a8+a14);
  a15=(a15*a25);
  a15=(a6*a15);
  a8=(a8+a15);
  a15=(a16*a8);
  a4=(a4-a15);
  if (res[1]!=0) res[1][0]=a4;
  a4=cos(a1);
  a15=(a10*a4);
  a15=(a3*a15);
  a25=sin(a1);
  a10=(a10*a25);
  a10=(a6*a10);
  a15=(a15-a10);
  a10=(a7*a15);
  a14=sin(a1);
  a14=(a3*a14);
  a1=cos(a1);
  a1=(a6*a1);
  a14=(a14+a1);
  a1=(a0*a14);
  a10=(a10-a1);
  a4=(a17*a4);
  a3=(a3*a4);
  a17=(a17*a25);
  a6=(a6*a17);
  a3=(a3-a6);
  a6=(a16*a3);
  a10=(a10+a6);
  if (res[1]!=0) res[1][1]=a10;
  a10=(a0*a2);
  a6=(a7*a11);
  a10=(a10+a6);
  a6=(a16*a18);
  a10=(a10+a6);
  a10=(-a10);
  if (res[1]!=0) res[1][2]=a10;
  a10=(a16*a20);
  a6=(a7*a13);
  a10=(a10-a6);
  if (res[1]!=0) res[1][3]=a10;
  a10=(a26*a19);
  a6=(a27*a8);
  a10=(a10-a6);
  if (res[1]!=0) res[1][4]=a10;
  a10=(a26*a15);
  a6=(a0*a14);
  a10=(a10-a6);
  a6=(a27*a3);
  a10=(a10+a6);
  if (res[1]!=0) res[1][5]=a10;
  a10=(a0*a2);
  a6=(a26*a11);
  a10=(a10+a6);
  a6=(a27*a18);
  a10=(a10+a6);
  a10=(-a10);
  if (res[1]!=0) res[1][6]=a10;
  a10=(a27*a20);
  a6=(a26*a13);
  a10=(a10-a6);
  if (res[1]!=0) res[1][7]=a10;
  a10=(a29*a19);
  a6=(a27*a8);
  a10=(a10-a6);
  if (res[1]!=0) res[1][8]=a10;
  a10=(a29*a15);
  a6=(a5*a14);
  a10=(a10-a6);
  a6=(a27*a3);
  a10=(a10+a6);
  if (res[1]!=0) res[1][9]=a10;
  a10=(a5*a2);
  a6=(a29*a11);
  a10=(a10+a6);
  a6=(a27*a18);
  a10=(a10+a6);
  a10=(-a10);
  if (res[1]!=0) res[1][10]=a10;
  a10=(a27*a20);
  a29=(a29*a13);
  a10=(a10-a29);
  if (res[1]!=0) res[1][11]=a10;
  a10=(a7*a19);
  a29=(a16*a8);
  a10=(a10-a29);
  if (res[1]!=0) res[1][12]=a10;
  a10=(a7*a15);
  a29=(a0*a14);
  a10=(a10-a29);
  a29=(a16*a3);
  a10=(a10+a29);
  if (res[1]!=0) res[1][13]=a10;
  a10=(a0*a2);
  a29=(a7*a11);
  a10=(a10+a29);
  a29=(a16*a18);
  a10=(a10+a29);
  a10=(-a10);
  if (res[1]!=0) res[1][14]=a10;
  a16=(a16*a20);
  a7=(a7*a13);
  a16=(a16-a7);
  if (res[1]!=0) res[1][15]=a16;
  a19=(a26*a19);
  a16=(a27*a8);
  a19=(a19-a16);
  if (res[1]!=0) res[1][16]=a19;
  a15=(a26*a15);
  a19=(a0*a14);
  a15=(a15-a19);
  a19=(a27*a3);
  a15=(a15+a19);
  if (res[1]!=0) res[1][17]=a15;
  a0=(a0*a2);
  a11=(a26*a11);
  a0=(a0+a11);
  a11=(a27*a18);
  a0=(a0+a11);
  a0=(-a0);
  if (res[1]!=0) res[1][18]=a0;
  a0=(a27*a20);
  a26=(a26*a13);
  a0=(a0-a26);
  if (res[1]!=0) res[1][19]=a0;
  a8=(a27*a8);
  a8=(-a8);
  if (res[1]!=0) res[1][20]=a8;
  a3=(a27*a3);
  a14=(a5*a14);
  a3=(a3-a14);
  if (res[1]!=0) res[1][21]=a3;
  a5=(a5*a2);
  a18=(a27*a18);
  a5=(a5+a18);
  a5=(-a5);
  if (res[1]!=0) res[1][22]=a5;
  a27=(a27*a20);
  if (res[1]!=0) res[1][23]=a27;
  return 0;
}

CASADI_SYMBOL_EXPORT int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int zyxEuler_kinematics_constr_h_e_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
