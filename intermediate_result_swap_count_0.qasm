OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];

// moment 0
h q[5];
h q[3];
h q[9];
h q[4];
h q[10];

// moment 1
cz q[9], q[5];

// moment 2
cz q[4], q[5];

// moment 3
cz q[4], q[3];

// moment 4
h q[11];
cz q[10], q[4];

// moment 5
cz q[10], q[9];
x q[5];
x q[4];

// moment 6
cz q[10], q[11];
x q[9];

// moment 7
cz q[11], q[3];
x q[10];

// moment 8
x q[3];
x q[11];

// measurement
measure q[5]->c[0];
measure q[3]->c[1];
measure q[11]->c[2];
measure q[9]->c[3];
measure q[4]->c[4];
measure q[10]->c[5];
