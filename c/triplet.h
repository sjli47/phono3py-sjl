/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* Some of these codes were originally parts of spglib, but only develped */
/* and used for phono3py. Therefore these were moved from spglib to */
/* phono3py. This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#ifndef __triplet_H__
#define __triplet_H__

#include <stddef.h>
#include "lagrid.h"

/* Irreducible triplets of k-points are searched under conservation of */
/* :math:``\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3 = \mathbf{G}``. */
/* Memory spaces of grid_address[prod(mesh)][3], map_triplets[prod(mesh)] */
/* and map_q[prod(mesh)] are required. rotations are point-group- */
/* operations in real space for which duplicate operations are allowed */
/* in the input. */
long tpl_get_triplets_reciprocal_mesh_at_q(long *map_triplets,
                                           long *map_q,
                                           long (*grid_address)[3],
                                           const long grid_point,
                                           const long mesh[3],
                                           const long is_time_reversal,
                                           const long num_rot,
                                           LAGCONST long (*rotations)[3][3],
                                           const long swappable);
/* Irreducible grid-point-triplets in BZ are stored. */
/* triplets are recovered from grid_point and triplet_weights. */
/* BZ boundary is considered in this recovery. Therefore grid addresses */
/* are given not by grid_address, but by bz_grid_address. */
/* triplets[num_ir_triplets][3] = number of non-zero triplets weights*/
/* Number of ir-triplets is returned. */
long tpl_get_BZ_triplets_at_q(long (*triplets)[3],
                              const long grid_point,
                              LAGCONST long (*bz_grid_address)[3],
                              const long *bz_map,
                              const long *map_triplets,
                              const long num_map_triplets,
                              const long mesh[3]);
void tpl_get_integration_weight(double *iw,
                                char *iw_zero,
                                const double *frequency_points,
                                const long num_band0,
                                LAGCONST long relative_grid_address[24][4][3],
                                const long mesh[3],
                                LAGCONST long (*triplets)[3],
                                const long num_triplets,
                                LAGCONST long (*bz_grid_address)[3],
                                const long *bz_map,
                                const double *frequencies1,
                                const long num_band1,
                                const double *frequencies2,
                                const long num_band2,
                                const long tp_type,
                                const long openmp_per_triplets,
                                const long openmp_per_bands);
void tpl_get_integration_weight_with_sigma(double *iw,
                                           char *iw_zero,
                                           const double sigma,
                                           const double sigma_cutoff,
                                           const double *frequency_points,
                                           const long num_band0,
                                           LAGCONST long (*triplets)[3],
                                           const long num_triplets,
                                           const double *frequencies,
                                           const long num_band,
                                           const long tp_type);

long tpl_is_N(const long triplet[3], const long *grid_address);
void tpl_set_relative_grid_address(
  long tp_relative_grid_address[2][24][4][3],
  LAGCONST long relative_grid_address[24][4][3],
  const long tp_type);

#endif
