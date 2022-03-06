import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd
import scipy.sparse.linalg as lg

vt_c = 0.3
sq2 = 2 ** 0.5
tu = 1
tv = 2
e_sq = 1 - (tu / tv) ** 2.0


def electromagnetic_anisotropic_dispersion(k, z, phi):
    # a = e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))
    # b = (e_sq * np.sin(phi) * np.cos(phi)) ** 2 / ((1 - e_sq * (np.cos(phi) ** 2)) * (1 - e_sq * (np.sin(phi) ** 2)))
    # b_factor = np.sqrt((1 - b) / (1 + b))
    a = -e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))

    # v_parallel
    # v_parallel = tu / np.sqrt(1 - e_sq * (np.sin(phi) ** 2))
    # v_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    # z_norm = np.sqrt(0.5 * (1 + b)) * z / v_parallel
    t_para = tu / np.sqrt((1 - e_sq) / (1 - e_sq * (np.cos(phi) ** 2)))
    t_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    z_norm = z / sq2 / t_para

    # Integrals
    # i1 = -0.25 * b_factor * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    # i2 = -0.25 * a * b_factor * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    # i3 = -0.25 * b_factor * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
    #                          2 * (1 + b) * (v_perp / v_parallel) ** 2 * pd.Zprime(z_norm))
    i1 = -0.25 * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    i2 = -0.25 * a * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    i3 = -0.25 * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
                  2 * (t_perp / t_para) ** 2 * pd.Zprime(z_norm))

    # Tensor components
    # t11 = 1 - (1 - i1) / ((k * z) ** 2)
    # t12 = i2 / ((k * z) ** 2)
    # t22 = 1 - 1 / ((vt_c * z) ** 2) - (1 - i3) / ((k * z) ** 2)
    t11 = z**2 - (1 - i1) / (k ** 2)
    t12 = i2 / (k ** 2)
    t22 = z**2 - 1 / (vt_c ** 2) - (1 - i3) / (k ** 2)

    # return -z ** 4 * (t11 * t22 - t12 ** 2)
    return -(t11 * t22 - t12 ** 2) / (z ** 2)


def electromagnetic_anisotropic_analytic_jacobian(k, z, phi):
    a = -e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))
    # b = (e_sq * np.sin(phi) * np.cos(phi)) ** 2 / ((1 - e_sq * (np.cos(phi) ** 2)) * (1 - e_sq * (np.sin(phi) ** 2)))
    # b_factor = np.sqrt((1 - b) / (1 + b))

    # v_parallel
    # v_parallel = tu / np.sqrt(1 - e_sq * (np.sin(phi) ** 2))
    # v_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    # z_norm = np.sqrt(0.5 * (1 + b)) * z / v_parallel
    t_para = tu / np.sqrt((1 - e_sq) / (1 - e_sq * (np.cos(phi) ** 2)))
    t_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    z_norm = z / sq2 / t_para

    # Integrals
    # i1 = -0.25 * b_factor * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    # i2 = -0.25 * a * b_factor * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    # i3 = -0.25 * b_factor * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
    #                          2 * (1 + b) * (v_perp / v_parallel) ** 2 * pd.Zprime(z_norm))
    i1 = -0.25 * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    i2 = -0.25 * a * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    i3 = -0.25 * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
                  2 * (t_perp / t_para) ** 2 * pd.Zprime(z_norm))
    # derivatives
    # di1 = -0.25 * b_factor * ((pd.Zquarticprime(z_norm) + 6 * pd.Zdoubleprime(z_norm)) *
    #                           np.sqrt((1 + b) / 2) / v_parallel)
    # # di2 = 0.25 * a * b_factor * ((pd.Zquarticprime(z_norm) + 8 * pd.Zdoubleprime(z_norm)) *
    # #                              np.sqrt((1 + b) / 2) / v_parallel)
    # di2 = -0.25 * a * b_factor * ((pd.Zquarticprime(z_norm) + 4 * pd.Zdoubleprime(z_norm)) *
    #                               np.sqrt((1 + b) / 2) / v_parallel)
    # di3 = -0.25 * b_factor * (((a ** 2) * (pd.Zquarticprime(z_norm) + 2 * pd.Zdoubleprime(z_norm)) +
    #                            2 * (1 + b) * (v_perp / v_parallel) ** 2 * pd.Zdoubleprime(z_norm)) *
    #                           np.sqrt((1 + b) / 2) / v_parallel)
    di1 = -0.25 * (pd.Zquarticprime(z_norm) + 6 * pd.Zdoubleprime(z_norm)) / sq2 / t_para
    di2 = -0.25 * a * (pd.Zquarticprime(z_norm) + 4 * pd.Zdoubleprime(z_norm)) / sq2 / t_para
    di3 = -0.25 * ((a ** 2) * (pd.Zquarticprime(z_norm) + 2 * pd.Zdoubleprime(z_norm)) +
                   2 * (t_perp / t_para) ** 2 * pd.Zdoubleprime(z_norm)) / sq2 / t_para

    # Tensor components
    # t11 = 1 - (1 - i1) / ((k * z) ** 2)
    # t12 = i2 / ((k * z) ** 2)
    # t22 = 1 - 1 / ((vt_c * z) ** 2) - (1 - i3) / ((k * z) ** 2)
    t11 = z ** 2 - (1 - i1) / (k ** 2)
    t12 = i2 / (k ** 2)
    t22 = z ** 2 - 1 / (vt_c ** 2) - (1 - i3) / (k ** 2)

    # d11 = 2 * (1 - i1) / ((k * z) ** 2) / z + di1 / ((k * z) ** 2)
    # d12 = -2 * i2 / ((k * z) ** 2) / z + di2 / ((k * z) ** 2)
    # d22 = 2 / ((z * vt_c) ** 2) / z + 2 * (1 - i3) / ((k * z) ** 2) / z + di3 / ((k * z) ** 2)
    d11 = 2 * z + di1 / (k ** 2)
    d12 = di2 / (k ** 2)
    d22 = 2 * z + di3 / (k ** 2)

    # return d11 * t22 + t11 * d22 - 2 * t12 * d12
    # return -z ** 4 * (d11 * t22 + t11 * d22 - 2 * t12 * d12) - 4 * (z ** 3) * (t11 * t22 - t12 ** 2)
    return -1*(-2 * (t11 * t22 - t12 ** 2) / (z ** 3) + (t11 * d22 + d11 * t22 - 2 * t12 * d12) / (z ** 2))


def dispersion_fsolve(z, k, phi):
    complex_z = z[0] + 1j * z[1]
    d = electromagnetic_anisotropic_dispersion(k, complex_z, phi)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, phi):
    complex_z = z[0] + 1j * z[1]
    jac = electromagnetic_anisotropic_analytic_jacobian(k, complex_z, phi)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


def eigenvalue_matrix(z, k, phi):
    # a = e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))
    # b = (e_sq * np.sin(phi) * np.cos(phi)) ** 2 / ((1 - e_sq * (np.cos(phi) ** 2)) * (1 - e_sq * (np.sin(phi) ** 2)))
    # b_factor = np.sqrt((1 - b) / (1 + b))
    a = -e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))

    # v_parallel
    # v_parallel = tu / np.sqrt(1 - e_sq * (np.sin(phi) ** 2))
    # v_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    # z_norm = np.sqrt(0.5 * (1 + b)) * z / v_parallel
    t_para = tu / np.sqrt((1 - e_sq) / (1 - e_sq * (np.cos(phi) ** 2)))
    t_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    z_norm = z / sq2 / t_para

    # Integrals
    # i1 = -0.25 * b_factor * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    # i2 = -0.25 * a * b_factor * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    # i3 = -0.25 * b_factor * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
    #                          2 * (1 + b) * (v_perp / v_parallel) ** 2 * pd.Zprime(z_norm))
    i1 = -0.25 * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    i2 = -0.25 * a * (pd.Ztripleprime(z_norm) + 4 * pd.Zprime(z_norm))
    i3 = -0.25 * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
                  2 * (t_perp / t_para) ** 2 * pd.Zprime(z_norm))

    # build matrix
    kle = k / vt_c  # electron inertial length
    matrix_a = np.array([[1 - i1, -i2], [-i2, kle ** 2 + 1 - i3]]) / (k ** 2)
    # print(a)
    # print('\nThe matrix chi is')
    # print(matrix_a)

    eigs = np.linalg.eig(matrix_a)
    # eigs = lg.eigs(matrix_a, k=2, tol=1e-12)
    # print(eigs)
    # quit()

    # print('\nComparison: ')
    # print(z)
    # print(np.sqrt(eigs[0][1]))
    # print(eigs[1][1])
    # quit()

    # pick correct eigenvalue
    for i in range(2):
        if np.abs(eigs[0][i] - z ** 2) < 1.0e-3:
            return eigs[1][i]


def rotation_matrix(angle):
    return [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]


if __name__ == '__main__':
    # phase velocities
    zr = np.linspace(-3, 3, num=400)
    zi = np.linspace(-5, 3, num=400)
    z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)
    ZR, ZI = np.meshgrid(zr, zi, indexing='ij')

    k_x = 0.05
    k_y = 0.0
    k = np.sqrt(k_x ** 2.0 + k_y ** 2.0)
    phi = np.arctan2(k_y, k_x)
    print('Angle is ' + str(360 * phi / (2 * np.pi)))
    ep = electromagnetic_anisotropic_dispersion(k, z, phi)

    plt.figure()
    plt.contour(ZR, ZI, np.real(ep), 0, colors='r', linewidths=3)
    plt.contour(ZR, ZI, np.imag(ep), 0, colors='g', linewidths=3)
    plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
    plt.axis('equal'), plt.grid(True), plt.tight_layout()

    solution = opt.root(dispersion_fsolve, x0=np.array([0, 0.75]),
                        args=(k, phi), jac=jacobian_fsolve, tol=1.0e-10)
    z = solution.x[0] + 1j * solution.x[1]

    print('Growth rate is ' + str(z*k))
    print('Square of z is ' + str(z ** 2))
    eigs = eigenvalue_matrix(z, k, phi)
    print('\neigs are ')
    # print(eigs)
    print('\nField vector angle is ')
    # print(np.arctan2(np.real(eigs[1]), np.real(eigs[0])) * 360 / (2 * np.pi))

    # Construct eigenmode
    # print(eigs[1][1])
    # rotated_ev = np.dot(rotation_matrix(angle=-phi), eigs)
    # print(rotated_ev)
    #
    # lx, ly = 2.0 * np.pi / k_x, 2.0 * np.pi / k_y
    # x, y = np.linspace(-lx / 2, lx / 2, num=100), np.linspace(-ly / 2, ly / 2, num=100)
    # X, Y = np.meshgrid(x, y, indexing='ij')
    # wave = np.exp(1j * (k_x * X + k_y * Y))
    #
    # Ex, Ey = np.real(rotated_ev[0] * wave), np.real(rotated_ev[1] * wave)
    # B = np.real((rotated_ev[1] - rotated_ev[0]) * wave / z)
    #
    # E = np.sqrt(Ex ** 2 + Ey ** 2)
    # cbe = np.linspace(np.amin(E), np.amax(E), num=100)
    # cbb = np.linspace(np.amin(B), np.amax(B), num=100)
    # EB_field_x = np.real((z * rotated_ev[1] / (rotated_ev[1] - rotated_ev[0])) * wave)
    # EB_field_y = np.real(-(z * rotated_ev[0] / (rotated_ev[1] - rotated_ev[0])) * wave)
    #
    # plt.figure()
    # cbex = np.linspace(np.amin(Ex), np.amax(Ex), num=100)
    # plt.contourf(X, Y, Ex, cbex)
    # plt.title(r'Electric-x E_x')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.colorbar(), plt.tight_layout()
    #
    # plt.figure()
    # cbey = np.linspace(np.amin(Ey), np.amax(Ey), num=100)
    # plt.contourf(X, Y, Ey, cbey)
    # plt.title(r'Electric-y E_y')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.colorbar(), plt.tight_layout()
    #
    # plt.figure()
    # plt.contourf(X, Y, E, cbe)
    # plt.title(r'Electric magnitude |E|')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.colorbar(), plt.tight_layout()
    #
    # plt.figure()
    # plt.contourf(X, Y, B, cbb)
    # plt.title(r'Magnetic field $B_z$')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.colorbar(), plt.tight_layout()
    #
    # plt.figure()
    # plt.streamplot(x, y, Ex.T, Ey.T)
    # plt.title(r'Electric streamlines $\vec{E} = E_x\hat{x} + E_y\hat{y}$')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.streamplot(x, y, EB_field_x.T, EB_field_y.T)
    # plt.title(r'ExB field, $\vec{E}\times\vec{B}/|B|^2$')
    # plt.xlabel('x'), plt.ylabel('y')
    # plt.tight_layout()

    plt.show()

    # Obtain some solutions
    kx = np.linspace(0.025, 0.55, num=20)
    ky = np.linspace(0, 0.15, num=40)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    sols = np.zeros_like(KX) + 0j
    guess_r, guess_i = 0, -1
    for idx, k_x in enumerate(kx):
        if idx > 0:
            guess_r, guess_i = np.real(sols[idx - 1, 0]), np.imag(sols[idx - 1, 0])
        for idy, k_y in enumerate(ky):
            k = np.sqrt(k_x ** 2.0 + k_y ** 2.0)
            phi = np.arctan2(k_y, k_x)
            # if phi > 1.3:  # np.abs(phi - np.pi/2) < 3.0e-1:
            #     continue
            solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                args=(k, phi), jac=jacobian_fsolve, tol=1.0e-10)
            # print(str(k_x) + ' ' + str(k_y) + ' ' + str(solution.x))
            guess_r, guess_i = solution.x
            sols[idx, idy] = (guess_r + 1j * guess_i)

    isol = np.imag(sols)
    cbi = np.linspace(np.amin(isol), np.amax(isol), num=100)
    plt.figure()
    plt.contourf(KX, KY, isol, cbi)  # , extend='both')
    plt.colorbar()
    plt.contour(KX, KY, isol, 0, colors='r')
    plt.xlabel(r'Wavenumber $(\vec{k}\lambda_D)\cdot\hat{e}_x$')
    plt.ylabel(r'Wavenumber $(\vec{k}\lambda_D)\cdot\hat{e}_y$')
    plt.title(r'Im($\zeta$)/$v_t$, $v_t/c=0.3$, $\theta_y/\theta_x=2$'), plt.tight_layout()
    plt.savefig('figs/vtc0p3_phase_velocity.pdf')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    K = np.sqrt(KX ** 2 + KY ** 2)
    om = K * isol
    cbo = np.linspace(np.amin(om), np.amax(om), num=100)
    plt.figure()
    plt.contourf(KX, KY, om, cbo)  # , extend='both')
    plt.colorbar()
    plt.contour(KX, KY, om, 0, colors='r')
    plt.xlabel(r'Wavenumber $(\vec{k}\lambda_D)\cdot\hat{e}_x$')
    plt.ylabel(r'Wavenumber $(\vec{k}\lambda_D)\cdot\hat{e}_y$')
    plt.title(r'Growth rate Im($\omega$)/$\omega_p$, $v_t/c=0.3$, $\theta_y/\theta_x=2$'), plt.tight_layout()
    plt.savefig('figs/vtc0p3_growth_rate.pdf')

    plt.show()
