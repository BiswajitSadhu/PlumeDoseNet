#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
from scipy import integrate, special
import logging
import time
from joblib import Parallel, delayed

class PLUMNESHINE:
    def __init__(self, device, config, logdir=None):
        self._device = device
        self.config = config
        self.logdir = logdir

        # Set defaults from config or use fallbacks
        self.sigma_z = None
        self.sigma_y = None
        self.rads_list = config.get('rads_list', [])
        self.sampling_time = config.get('sampling_time', 60)
        self.release_height = config.get('release_height', 10.0)
        self.measurement_height = config.get('measurement_height', 1.0)
        self.spatial_distance = config.get('spatial_distance', 100)
        self.stability_category = config.get('stability_category', '2')

    # (All other methods remain unchanged and are preserved as-is)
    
            
    def sigmay(self, stab_cat, X1):
        """
        Compute the standard deviation in the y-direction (sigmay) based on stability category and distance.

        Args:
        - stab_cat (int): Stability category (1-6).
        - X1 (float): Downwind distance (in meters).

        Returns:
        - SIGY (float): Standard deviation in the y-direction.
        - AY (float): Coefficient AY for the stability category.

        Notes:
        - This method computes the standard deviation in the y-direction (sigmay) based on the stability category and the downwind distance.
        - The standard deviation is calculated using the formula: SIGY = AY * (X1 ** 0.9031), where AY is a coefficient specific to the stability category.
        - The stability category ranges from 1 to 6, with 1 being the least stable and 6 being the most stable.
        - The downwind distance X1 should be provided in meters.
        - The sampling time is used to apply a correction factor (CF) to the standard deviation calculation, if applicable.
        - The calculated SIGY and AY values are returned as a tuple.

        """
        ay_list = [0.3658, 0.2751, 0.2089, 0.1471, 0.1046, 0.0722]
        AY = ay_list[stab_cat - 1]
        SIGY = AY * (X1 ** 0.9031)
        # SIGY = SIGY*CF
        # self.sigma_y = SIGY
        # self.AY = AY
        return SIGY, AY

    def sigmaz(self, stab_cat, X):
        """
            Calculate sigma-Z (SIGZ) for atmospheric dispersion modeling.

            Parameters:
                stab_cat (int): Stability category representing atmospheric stability.
                X (float): Downwind distance parameter.

            Returns:
                tuple: A tuple containing SIGZ (sigma-Z), AZ, Q, and R.
                    - SIGZ (float): Sigma-Z value calculated for the given parameters.
                    - AZ (float): Coefficient for the stability category.
                    - Q (float): Coefficient for the stability category.
                    - R (float): Coefficient for the stability category.

            Raises:
                ValueError: If the downwind distance (X) is not a number.
        """
        X = float(X)
        if X < 100:
            az_list = [0.192, 0.156, 0.116, 0.079, 0.063, 0.053]
            q_list = [0.936, 0.922, 0.905, 0.881, 0.871, 0.814]
            r_list = [0, 0, 0, 0, 0, 0]
            AZ = az_list[stab_cat - 1]
            Q = q_list[stab_cat - 1]
            R = r_list[stab_cat - 1]

        elif 100 <= X <= 1000:
            az_list = [0.00066, 0.038, 0.113, 0.222, 0.211, 0.086]
            q_list = [1.941, 1.149, 0.911, 0.725, 0.678, 0.740]
            r_list = [9.27, 3.3, 0, -1.7, -1.3, -0.35]
            AZ = az_list[stab_cat - 1]
            Q = q_list[stab_cat - 1]
            R = r_list[stab_cat - 1]

        elif X > 1000:
            az_list = [0.00024, 0.055, 0.113, 1.26, 6.73, 18.05]
            q_list = [2.094, 1.098, 0.911, 0.516, 0.305, 0.180]
            r_list = [-9.6, 2.0, 0, -13.0, -34.0, -48.6]
            AZ = az_list[stab_cat - 1]
            Q = q_list[stab_cat - 1]
            R = r_list[stab_cat - 1]

        else:
            raise ValueError('The downwind distance must be a number.')

        SIGZ = (AZ * (X ** Q)) + R
        # self.sigma_z = SIGZ
        return SIGZ, AZ, Q, R
    
    def master_eq_single_plume(self, SIGMAY, SIGMAZ, factor):
        """
            Calculate the concentration of pollutants using the master equation for a single plume.

            The master equation is taken from Hukko and Bapat's work (eq. 2.5) and calculates the concentration
            of pollutants at a given point based on the input parameters.

            Parameters:
                SIGMAY (float): Standard deviation of the plume in the lateral (crosswind) direction.
                SIGMAZ (float): Standard deviation of the plume in the vertical (height) direction.
                factor (float): Factor to scale the speed of dispersion.

            Returns:
                tuple: A tuple containing pre_expo and expo.
                    - pre_expo (float): Pre-exponential factor in the master equation.
                    - expo (float): Exponential term in the master equation.

            Notes:
                By default, if the configuration is set to maximize concentration along the plume's central line
                on the ground, Y and Z are set to 0. Otherwise, the user can provide custom values for Y and Z.
                Speed is assumed to be unit speed and can be scaled using the factor parameter.

        """
        
        Y = 0
        Z = 0

        # Assuming unit speed and scaling with factor
        speed = 1
        speed = speed * factor
        pre_expo = 1 / (2 * np.pi * SIGMAY * SIGMAZ * speed)
        expo = (np.exp(-1 * np.divide(np.square(Y), (2 * np.square(SIGMAY))))) * \
               (np.exp(-1 * np.divide(np.square(Z - self.release_height), (2 * np.square(SIGMAZ))))
                + np.exp(-1 * np.divide(np.square(Z + self.release_height), (2 * np.square(SIGMAZ)))))

        return pre_expo, expo
        
    def add_zero_energy_for_pure_beta(self, energies, emission_prob):
        """
            Add zero energy and emission probability for pure beta emitters or cases where gamma energy is not available.

            This method adds a zero energy value and emission probability for pure beta emitters or cases where gamma energy
            is not available in the data. It modifies the energies and emission probabilities lists to include the zero
            energy value.

            Args:
                energies (list): List of gamma energies for radionuclides.
                emission_prob (list): List of corresponding emission probabilities for the gamma energies.

            Returns:
                tuple: A tuple containing two lists:
                    - Modified list of gamma energies, including zero energy where necessary.
                    - Modified list of emission probabilities, including zero values where necessary.
        """
        # unit=photons/m2-s
        # integral_stab_cat_energy_wise = []
        # empty list for pure-beta emitter or not-available gamma is filled with [0]
        energies_mod = []
        emission_prob_mod = []
        for i, j in zip(energies, emission_prob):
            if not i:
                i = [0]
                j = [0]
            emission_prob_mod.append(j)
            energies_mod.append(i)
        return energies_mod, emission_prob_mod
    
    def gamma_energy_abundaces(self, master_file="./Library/Dose_ecerman_final.xlsx",
                               sheet_name="gamma_energy_radionuclide"):
        """
            Extract gamma energy and abundances data from a specified database.

            This method retrieves gamma energy and emission probability data from a specified database and returns them as lists
            along with a dictionary containing the energy-emission probability pairs. It also identifies and logs neglected
            energies based on certain cutoff criteria. Taken from following database:
            https://www-nds.iaea.org/xgamma_standards/genergies1.htm

            Args:
                master_file (str, optional): Path to the master Excel file containing the data. Defaults to library/Dose_ecerman_final.xlsx".
                sheet_name (str, optional): Name of the sheet in the Excel file containing the data. Defaults to "gamma_energy_radionuclide".

            Returns:
                tuple: A tuple containing:
                    - energies (list): List of lists containing gamma energies for each radionuclide.
                    - emission_prob (list): List of lists containing emission probabilities for each radionuclide.
                    - en_dict (dict): Dictionary containing energy-emission probability pairs.
                    - neglected_energies (dict): Dictionary containing neglected energies based on cutoff criteria.

        """
        logging.getLogger("main").info("Data source of gamma energy and abundances: {weblk}".format(
            weblk="www-nds.iaea.org/xgamma_standards/genergies1.htm"))
        xls = pd.ExcelFile(master_file)
        colnames = ['nuclide', 'energy_kev', 'std_energy_kev', 'emmission_prob', 'std_emmission_prob', '']
        name = pd.read_excel(xls, sheet_name, header=None, names=colnames)
        name.dropna(axis=0, how='all', inplace=True)
        name = name.iloc[:, :-1]
        energies = []
        emmission_prob = []
        en_dict = {}
        neglected_energies = {}
        for rad in self.rads_list:
            energies_per_rad = []
            emmission_prob_per_rad = []
            search_string = '|'.join([rad])
            df = name[name['nuclide'].str.contains(search_string, na=False)]
            if df.empty:
                emmission_prob_per_rad = [0]
                energies_per_rad = [0]
                print("gamma energy not available for {}. This either means the radionuclide is pure-beta emitter or "
                      "the data of gamma energy not available in the current database (ref: www-nds.iaea.org/xgamma_standards/genergies1.htm)".format(
                    rad))

                logging.getLogger("gamma energies").info(
                    "gamma energy not available for {}. This either means the radionuclide is pure-beta emitter or "
                    "the data of gamma energy not available in the current database (ref: www-nds.iaea.org/xgamma_standards/genergies1.htm)".format(
                        rad))

            df_e = df['energy_kev'].items()
            df_p = df['emmission_prob'].items()
            for (column_e, content_e), (c, content_p) in zip(df_e, df_p):
                # neglected based on cutoff criteria (below 5 kev or abundance below 1e-03)
                if content_e / 1000 < 0.05 or content_p < 0.001:
                    # converted to MeV unit
                    neglected_energies[rad] = (content_e / 1000, content_p)
                # above 5 kev and abundance above 1e-03
                else:
                    energies_per_rad.append(content_e / 1000)
                    emmission_prob_per_rad.append(content_p)
                    en_dict[content_e / 1000] = content_p

                # converted to MeV unit
                # energies_per_rad.append(content_e / 1000)
                # emmission_prob_per_rad.append(content_p)
                # en_dict[content_e / 1000] = content_p
            energies.append(energies_per_rad)
            emmission_prob.append(emmission_prob_per_rad)
        return energies, emmission_prob, en_dict, neglected_energies
    
    def height_correction_factor(self, stab_cat):
        """
            Compute the height correction factor based on the stability category.

            Parameters:
                stab_cat (int): Stability category (1 to 6).

            Returns:
                float: Height correction factor.

            Raises:
                ValueError: If stability category is not an integer from 1 to 6.

            Notes:
                The height correction factor adjusts for the difference in release height and measurement height
                in the atmospheric dispersion calculations. It depends on the stability category of the atmosphere.
                If the release height is less than 10 meters, it's set to 10 meters to avoid dividing by zero or negative values.
                Then, the height correction factor is computed using the adjusted release height and the measurement height.
                If the release height is greater than or equal to 10 meters, the factor is computed directly using the actual release height.

            References:
                - Pasquill, F. (1974). Atmospheric Diffusion (3rd ed.). Horwood.

        """
        if stab_cat < 4:
            AN = 0.2
        elif stab_cat == 4:
            AN = 0.25
        elif stab_cat > 4:
            AN = 0.50
        else:
            raise ValueError('Stability category must be an integer (1 to 6).')

        P = AN / (2.0 - AN)

        if self.release_height < 10:
            release_height = 10
            factor = (release_height / self.measurement_height) ** P
        else:
            factor = (self.release_height / self.measurement_height) ** P
        return factor

    
    def get_k_mu_mua_MFP(self, energies):
        """
            Calculate various parameters related to attenuation for a list of energies.

            This function calculates the mass attenuation coefficient (mu), mass energy-absorption coefficient (mu_a),
            mass scattering coefficient (k), and mean free path (MFP) for a list of energies. The values are obtained
            by interpolating data from an Excel file.

            Args:
                energies (list): List of energies in MeV.

            Returns:
                dict: A dictionary containing the calculated parameters for each energy.
        """
        k_mu_mua_MFP_dict = {}
        energies = sum(energies, [])
        for energy in energies:
            mu, mu_a = self.atten_coeff(master_file="./Library/Dose_ecerman_final.xlsx",
                                        sheet_name="mass_attenuation_coeff",
                                        energy_rad=[energy])
            k = (mu - mu_a) / mu_a
            MFP = 1 / mu
            k_mu_mua_MFP_dict[energy] = (k, mu, mu_a, MFP)
        for key, val in k_mu_mua_MFP_dict.items():
            logging.getLogger("plume_shine").info(
                "For energy {}, values of k={}, mu={}, mua={} and MFP={}'.format(key, val[0], val[1], val[2], val[3])")
            print('For energy {}, values of k={}, mu={}, mua={} and MFP={}'.format(key, val[0], val[1], val[2], val[3]))
        return k_mu_mua_MFP_dict
    
    def get_limit_lists_per_rad_for_all_energies(self, k_mu_mua_MFP_dict, X1, all_energy_per_rad):
        """
            Calculate limit lists for each energy and stability category combination.

            This function calculates the limit lists for each energy and stability category combination. It iterates over
            all energies and stability categories, obtaining the appropriate parameters from a dictionary, and then calls
            either the zyx_lim_for_integral_sector_averaged_plume or zyx_lim_for_integral_single_plume method to get the
            limit lists.

            Args:
                k_mu_mua_MFP_dict (dict): A dictionary containing parameters calculated for each energy.
                X1 (float): Spatial distance.
                all_energy_per_rad (list): List of energies for each radionuclide.

            Returns:
                numpy.ndarray: A 3D numpy array containing the limit lists for each energy and stability category combination.
                               The shape of the array is (num_energies, 6, 6).
        """
        all_limit_lists_per_rad = []
        for energy in all_energy_per_rad:
            # MFP is obtained using average energy of gamma-emitter; y limit is different in sect-av
            k, mu, mua, MFP = k_mu_mua_MFP_dict[energy]
            for j in np.arange(0, 6, 1):
                stab_cat = j + 1
                limit_list = self.zyx_lim_for_integral_single_plume(stab_cat, MFP)
                all_limit_lists_per_rad.append(limit_list)
        # make a stack of limit list for six stability categories for each energy. each limit list
        # contains six values
        limit_list_energy_wise = np.array(all_limit_lists_per_rad).reshape(len(all_energy_per_rad), 6, 6)
        return limit_list_energy_wise
    
    def atten_coeff(self, master_file='./Library/Dose_ecerman_final.xlsx', sheet_name='mass_attenuation_coeff',
                    energy_rad=None):
        """
            Calculate attenuation coefficients for radiation in air.

            This method calculates attenuation coefficients for radiation in air based on the energy of the radiation.
            It retrieves attenuation coefficient data from the specified Excel file and interpolates the coefficients
            based on the given energy of the radiation. taken from NIST database for air medium:
            https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html.

            Args:
                master_file (str): The path to the Excel file containing mass attenuation coefficient data.
                sheet_name (str): The name of the sheet in the Excel file containing the mass attenuation coefficient data.
                energy_rad (float): The energy of the radiation in MeV.

            Returns:
                tuple: A tuple containing the total attenuation coefficient and the energy-dependent attenuation coefficient
                for the given radiation energy. Both coefficients are multiplied by air density in g/cm^3 and 100 to convert
                the unit to /m.

        """
        energy_rad = energy_rad
        xls = pd.ExcelFile(master_file)
        colnames = ['energy', 'total_atten_coeff', 'energy_atten_coeff']
        df = pd.read_excel(xls, sheet_name, names=colnames)
        # energy in MeV; total_atten_coeff and energy_atten_coeff in /m (cm2/g*gm/cm3*100)
        total_atten_coeff = np.interp(energy_rad, df['energy'], df['total_atten_coeff']) * 1.225E-3 * 100
        energy_atten_coeff = np.interp(energy_rad, df['energy'], df['energy_atten_coeff']) * 1.225E-3 * 100
        return total_atten_coeff[0], energy_atten_coeff[0]
    
    def zyx_lim_for_integral_single_plume(self, stab_cat, MFP):
        X1 = float(self.spatial_distance)
        SIGMAZ = self.sigmaz(stab_cat, X1)[0]
        SIGMAY = self.sigmay(stab_cat, X1)[0]
        logging.getLogger("main").info(
            "Sigma Y: {sigmay}, Sigma Z: {sigmaz}, Stability Category: {sc}".format(sigmay=SIGMAY, sigmaz=SIGMAZ,
                                                                                    sc=stab_cat))

        # find limits for triple integral evaluation
        if self.release_height - (3 * SIGMAZ) > 1:
            zin_lim = self.release_height - (3 * SIGMAZ)
        else:
            zin_lim = 1
        zf_lim = self.release_height + (3 * SIGMAZ)
        yin_lim = -3 * SIGMAY
        yf_lim = 3 * SIGMAY
        # Cite (for MFP): Wang, X. Y., Y. S. Ling, and Z. Q. Shi. "A new finite cloud method for calculating
        # external exposure dose in a nuclear emergency." Nuclear engineering and design 231, no. 2 (2004): 211-216.
        # to solve the problem of singularity the integration is terminated at 1 m downwind distance from
        # the release point
        times = 3
        if X1 - (times * MFP) > 1:
            xin_lim = X1 - (times * MFP)
        else:
            xin_lim = 1
        xf_lim = X1 + (times * MFP)
        logging.getLogger("Spatial Limits for Plume Dose Calculation").info(
            "zin_lim: {zin_lim}, zf_lim: {zf_lim}, yin_lim: {yin_lim}, yf_lim: {yf_lim}, xin_lim: {xin_lim}, "
            "xf_lim: {xf_lim}" \
                .format(zin_lim=zin_lim, zf_lim=zf_lim, yin_lim=yin_lim, yf_lim=yf_lim, xin_lim=xin_lim,
                        xf_lim=xf_lim))

        return np.array([zin_lim, zf_lim, yin_lim, yf_lim, xin_lim, xf_lim])

    def plumeshine_dose(self):
        X1 = self.spatial_distance
        WSPEED_K = np.array([0.9, 2.4, 4.25, 8.5, 15.5, 24.5, 34.0, 44.5, 56.0, 68.0]) / 3.6
        SECWID = 0.39275

        energies, emission_prob, en_dict, neglected_energies = self.gamma_energy_abundaces(
            master_file="./Library/Dose_ecerman_final.xlsx",
            sheet_name="gamma_energy_radionuclide")

        energies, emission_prob = self.add_zero_energy_for_pure_beta(energies, emission_prob)
        k_mu_mua_MFP_dict = self.get_k_mu_mua_MFP(energies)
        height_factors = [self.height_correction_factor(sc) for sc in np.arange(1, 7, 1)]

        def adgq_single_plume(ndx, energy, limit_list, k_mu_mua_MFP_dict):
            stab_cat = ndx + 1
            limit_list = limit_list[ndx]
            X1 = self.spatial_distance
            k = k_mu_mua_MFP_dict[energy][0]
            mu = k_mu_mua_MFP_dict[energy][1]
            Y = 0
            Z = 0

            expo_xyz = lambda x, y, z: (1 / (
                    2 * np.pi * self.sigmay(stab_cat, x)[0] * self.sigmaz(stab_cat, x)[0])) * \
                                       (((1 + (k * mu * np.sqrt((x - X1) ** 2 + (y - Y) ** 2 + (z - Z) ** 2)))
                                         / (4 * np.pi * ((x - X1) ** 2 + (y - Y) ** 2 + (z - Z) ** 2)))) * \
                                       (np.exp(-mu * np.sqrt((x - X1) ** 2 + (y - Y) ** 2 + (z - Z) ** 2))) * \
                                       (np.exp((-0.5 * y ** 2) / (self.sigmay(stab_cat, x)[0] ** 2))) * \
                                       (np.exp(-0.5 * (z - self.release_height) ** 2 / (self.sigmaz(stab_cat, x)[0] ** 2)) +
                                        np.exp(-0.5 * (z + self.release_height) ** 2 / (self.sigmaz(stab_cat, x)[0] ** 2)))

            int_expo_xyz = integrate.tplquad(expo_xyz, limit_list[0], limit_list[1], limit_list[2],
                                             limit_list[3], limit_list[4], limit_list[5], epsabs=1.49e-02,
                                             epsrel=1.49e-02)
            return int_expo_xyz[0]

        def get_all_integral_stab_cat_energy_wise_for_all_rad_parallel(X1, energies):
            all_integral_stab_cat_energy_wise = []
            for all_energy_per_rad in energies:
                limit_list_energy_wise = self.get_limit_lists_per_rad_for_all_energies(
                    k_mu_mua_MFP_dict, X1, all_energy_per_rad)
                integral_stab_cat_energy_wise = []
                for edx, energy in enumerate(all_energy_per_rad):
                    limit_list_per_energy = limit_list_energy_wise[edx]
                    results = Parallel(n_jobs=-1, verbose=100)(
                        delayed(adgq_single_plume)(ndx, energy, limit_list_per_energy, k_mu_mua_MFP_dict)
                        for ndx in range(6))
                    integral_stab_cat_energy_wise.append(list(results))
                all_integral_stab_cat_energy_wise.append(integral_stab_cat_energy_wise)
            return all_integral_stab_cat_energy_wise

        t = time.time()
        all_integral_stab_cat_energy_wise = get_all_integral_stab_cat_energy_wise_for_all_rad_parallel(X1, energies)
        print('results of parallel plume shine integrals:', all_integral_stab_cat_energy_wise)
        print('time_taken_in_plume_shine_dose_computation:', time.time() - t)

        pl_sh_sectors_list = []
        for raddx in range(len(energies)):
            pl_sh_sectors = []
            for ndx, (energy_rad, abundance_rad) in enumerate(zip(energies[raddx], emission_prob[raddx])):
                mu_a = k_mu_mua_MFP_dict[energy_rad][2]
                for i in np.arange(0, 6, 1):
                    plumeshine_dose = 5e-4 * energy_rad * mu_a * \
                                      all_integral_stab_cat_energy_wise[raddx][ndx][i] * abundance_rad
                    pl_sh_sectors.append(plumeshine_dose)
            pl_sh_sectors = np.array(pl_sh_sectors, dtype=object).reshape(-1, 6).sum(axis=0)
            pl_sh_sectors_list.append(pl_sh_sectors)

        pl_sh_sectors_list_all_year = np.array(pl_sh_sectors_list)[:, None]
        pl_sh_sectors_list_all_year = [[[b * x for x in inner] for inner in outer] for b, outer in
                                       zip(self.config['instantaneous_release_bq_list'],
                                           pl_sh_sectors_list_all_year)]
        
        stab_class_map ={'1':0, '2':1, '3':2, '4':3, '5':4, '6':5}
        stab_index = stab_class_map[self.stability_category]
        ps_dose = np.array(pl_sh_sectors_list_all_year).flatten()[stab_index]
        
        return ps_dose
