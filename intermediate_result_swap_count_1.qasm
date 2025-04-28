OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];

// moment 0
h q[10];
h q[4];
h q[11];
h q[3];
cx q[1], q[13];

// moment 1
cz q[4], q[10];
cx q[13], q[1];

// moment 2
h q[12];
cz q[11], q[10];
cx q[1], q[13];

// moment 3
cz q[11], q[12];

// moment 4
cz q[3], q[11];

// moment 5
h q[2];
cz q[3], q[4];
x q[11];

// moment 6
cz q[3], q[2];

// moment 7
cz q[2], q[12];
x q[4];
x q[3];

// moment 8
x q[10];
x q[12];
x q[2];

// measurement
measure q[10]->c[0];
measure q[12]->c[1];
measure q[2]->c[2];
measure q[4]->c[3];
measure q[11]->c[4];
measure q[3]->c[5];
