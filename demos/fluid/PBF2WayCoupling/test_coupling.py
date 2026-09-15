"""Independent formula and rigid-body checks; CPU by default, CUDA parity if available."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp

from . import PBF2WayCoupling as solver
from . import coupling_initialization as init
from . import coupling_functions as fn


@wp.kernel
def evaluate_kernels(x: wp.array(dtype=wp.vec3), h: float, values: wp.array(dtype=float), gradients: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    values[i] = fn.poly6(wp.length(x[i]), h)
    gradients[i] = fn.spiky_gradient(x[i], h)


@wp.kernel
def query_distances(mesh: wp.uint64, x: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, x[i], 10.0)
    p = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    distances[i] = q.sign * wp.length(p-x[i])


def w(r, h):
    return 315/(64*np.pi*h**9)*max(h*h-r*r, 0)**3


def grad(x, h):
    r = np.linalg.norm(x)
    return -45/(np.pi*h**6)*(h-r)**2*x/r if 1e-9 < r < h else np.zeros(3)


class CouplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.scene = Path(cls.temp.name)/"small.json"
        assets = init.ROOT / "assets"
        cls.scene.write_text(json.dumps({"RigidBodies": [
            {"geometryFile": str(assets/"UnitBox.obj"), "scale": [.8,1,.8], "translation": [0,.5,0], "isWall": True},
            {"geometryFile": str(assets/"sphere.obj"), "scale": [.12]*3, "translation": [.12,.35,0], "density": 500, "isDynamic": True},
        ], "FluidBlocks": [{"start": [-.3,0,-.3], "end": [.3,.6,.3]}]}))
        cls.config = solver.PBF2WayCouplingConfig(scene=str(cls.scene), particle_radius=.05)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make(self, **overrides):
        return solver.Example(replace(self.config, **overrides), device="cpu")

    def search(self, s):
        s.fluid_grid.build(s.positions, s.support_radius)
        s.boundary_grid.build(s.boundary.position, s.support_radius)
        wp.launch(solver.cache_neighbors, s.num_particles,
                  [s.fluid_grid.id,s.boundary_grid.id,s.positions,s.boundary,s.support_radius,s.neighbors], device=s.device)

    def test_obj_mass_inertia_and_sampling(self):
        v, f = init.load_obj(init.ROOT/"assets/UnitBox.obj")
        v = v * [2,3,4] + [5,-2,7]
        mass, center, inertia = init.mass_properties(v, f, 10)
        self.assertAlmostEqual(mass, 240, places=9)
        np.testing.assert_allclose(center, [5,-2,7], atol=1e-12)
        np.testing.assert_allclose(inertia, np.diag([500,400,260]), atol=1e-9)
        v, f = init.load_obj(init.ROOT/"assets/sphere.obj")
        mass, center, inertia = init.mass_properties(v*.2, f, 1000)
        self.assertAlmostEqual(mass/(1000*4*np.pi*.2**3/3), 1, delta=.03)
        np.testing.assert_allclose(np.diag(inertia), np.full(3, .4*mass*.2**2), rtol=.03)
        samples = init.sample_surface(v, f, .2)
        np.testing.assert_array_equal(samples, init.sample_surface(v, f, .2))
        self.assertEqual(len(np.unique(np.round(samples,5), axis=0)), len(samples))

    def test_kernel_origin_support_edge_and_outside(self):
        s = self.make()
        x = np.array([[0,0,0],[.03,.02,.01],[.199999,0,0],[.2,0,0],[.3,0,0]],dtype=np.float32)
        values = wp.zeros(len(x),dtype=float,device="cpu")
        gradients = wp.zeros(len(x),dtype=wp.vec3,device="cpu")
        wp.launch(evaluate_kernels,len(x),[wp.array(x,dtype=wp.vec3,device="cpu"),.2,values,gradients],device="cpu")
        np.testing.assert_allclose(values.numpy(),[w(np.linalg.norm(a.astype(float)),.2) for a in x],rtol=1e-5,atol=1e-9)
        np.testing.assert_allclose(gradients.numpy(),[grad(a.astype(float),.2) for a in x],rtol=1e-5,atol=1e-7)

    def test_torus_hole_is_free_space(self):
        self.make()  # Initialize Warp and its local cache.
        v,f = init.load_obj(init.ROOT/"assets/torus.obj")
        mesh = wp.Mesh(wp.array(v*.2,dtype=wp.vec3,device="cpu"),wp.array(f.flatten(),dtype=int,device="cpu"))
        # The official torus lies in XZ, with major radius 1 and tube radius .5.
        points = wp.array([[0,0,0],[.2,0,0],[.4,0,0]],dtype=wp.vec3,device="cpu")
        distances = wp.zeros(3,dtype=float,device="cpu")
        wp.launch(query_distances,3,[mesh.id,points,distances],device="cpu")
        self.assertGreater(distances.numpy()[0],.09)
        self.assertLess(distances.numpy()[1],-.09)
        self.assertGreater(distances.numpy()[2],.09)

    def test_standard_viscosity_and_reaction_against_numpy(self):
        s = self.make(boundary_viscosity=.04)
        rng = np.random.default_rng(7)
        vel = rng.normal(0,.2,(s.num_particles,3)).astype(np.float32)
        s.velocities.assign(vel)
        self.search(s)
        wp.launch(solver.density_lambda,s.num_particles,[s.positions,s.boundary,s.neighbors,s.fluid_volume,s.support_radius,s.densities,s.lambdas,s.error],device="cpu")
        wp.launch(solver.viscosity,s.num_particles,[s.positions,s.velocities,s.boundary,s.rigid,s.neighbors,s.densities,s.fluid_volume,s.fluid_mass,s.support_radius,.01,.04,1,s.acceleration],device="cpu")
        x = s.positions.numpy().astype(float)
        bx = s.boundary.position.numpy().astype(float)
        bv = s.boundary.volume.numpy()
        density = s.densities.numpy()
        ids = s.boundary.body.numpy()
        acc = np.zeros_like(x)
        force = np.zeros((2,3))
        h = s.support_radius
        for i,xi in enumerate(x):
            for j in np.flatnonzero(np.linalg.norm(x-xi,axis=1)<h):
                if i != j:
                    delta = xi-x[j]
                    acc[i] += 10*.01*s.fluid_volume/density[j]*np.dot(vel[i]-vel[j],delta)/(delta@delta+.01*h*h)*grad(delta,h)
            for j in np.flatnonzero(np.linalg.norm(bx-xi,axis=1)<h):
                delta = xi-bx[j]
                a = 10*.04*bv[j]/density[i]*np.dot(vel[i],delta)/(delta@delta+.01*h*h)*grad(delta,h)
                acc[i] += a
                if ids[j] == 1:
                    force[1] -= s.fluid_mass*a
        np.testing.assert_allclose(s.acceleration.numpy(),acc,rtol=1e-5,atol=3e-6)
        np.testing.assert_allclose(s.rigid.force.numpy(),force,rtol=2e-5,atol=3e-6)

    def test_density_lambda_correction_and_force_against_brute_force(self):
        s = self.make()
        # Compress the cloud to produce positive density constraints.
        points = s.positions.numpy()
        points[:,0] *= .6
        s.positions.assign(points)
        self.search(s)
        self.assertFalse(s.neighbors.overflow.numpy().any())
        wp.launch(solver.density_lambda, s.num_particles,
                  [s.positions,s.boundary,s.neighbors,s.fluid_volume,s.support_radius,s.densities,s.lambdas,s.error], device=s.device)
        x, bx, bv = points.astype(float), s.boundary.position.numpy().astype(float), s.boundary.volume.numpy().astype(float)
        h, volume, mass = s.support_radius, s.fluid_volume, s.fluid_mass
        densities, lambdas = [], []
        for i, xi in enumerate(x):
            rho = volume*w(0,h)
            gi, denominator = np.zeros(3), 0.
            for j in np.flatnonzero(np.linalg.norm(x-xi,axis=1)<h):
                if i == j:
                    continue
                rho += volume*w(np.linalg.norm(xi-x[j]),h)
                gj = -volume*grad(xi-x[j],h)
                gi -= gj
                denominator += gj@gj
            for j in np.flatnonzero(np.linalg.norm(bx-xi,axis=1)<h):
                rho += bv[j]*w(np.linalg.norm(xi-bx[j]),h)
                gi += bv[j]*grad(xi-bx[j],h)
            densities.append(rho)
            lambdas.append(-max(rho-1,0)/(denominator+gi@gi+1e-6))
        np.testing.assert_allclose(s.densities.numpy(), densities, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(s.lambdas.numpy(), lambdas, rtol=3e-5, atol=1e-8)
        self.assertGreater(max(densities), 1.05)
        correction = np.zeros_like(x)
        forces, torques = np.zeros((2,3)), np.zeros((2,3))
        bodies = s.boundary.body.numpy()
        centers = s.rigid.position.numpy()
        for i, xi in enumerate(x):
            for j in np.flatnonzero(np.linalg.norm(x-xi,axis=1)<h):
                if i != j:
                    correction[i] += (lambdas[i]+lambdas[j])*volume*grad(xi-x[j],h)
            for j in np.flatnonzero(np.linalg.norm(bx-xi,axis=1)<h):
                dx = lambdas[i]*bv[j]*grad(xi-bx[j],h)
                correction[i] += dx
                b = bodies[j]
                if b == 1:
                    force = -mass*dx/s.current_dt**2
                    forces[b] += force
                    torques[b] += np.cross(bx[j]-centers[b],force)
        wp.launch(solver.pressure_correction, s.num_particles,
                  [s.positions,s.boundary,s.rigid,s.neighbors,volume,mass,h,s.current_dt,1,s.lambdas,s.corrections], device=s.device)
        np.testing.assert_allclose(s.corrections.numpy(), correction, rtol=2e-4, atol=1e-7)
        np.testing.assert_allclose(s.rigid.force.numpy(), forces, rtol=1e-4, atol=.1)
        np.testing.assert_allclose(s.rigid.torque.numpy(), torques, rtol=1e-4, atol=.02)

    def test_boundary_volume_groups_and_rigid_transformation(self):
        s = self.make()
        bx, group = s.boundary.position.numpy(), s.boundary.body.numpy()
        volume = s.boundary.volume.numpy()
        for i in [0,100,len(bx)-1]:
            expected = 1/sum(w(float(np.linalg.norm(bx[i]-bx[j])),s.support_radius) for j in np.flatnonzero(group==group[i]))
            self.assertAlmostEqual(volume[i]/expected, 1, delta=2e-5)
        q = init.rotation_quaternion([0,1,0], .7)
        s.rigid.rotation.assign(np.array([[0,0,0,1],q], dtype=np.float32))
        s.rigid.velocity.assign(np.array([[0,0,0],[1,2,3]],dtype=np.float32))
        s.rigid.omega.assign(np.array([[0,0,0],[0,2,0]],dtype=np.float32))
        wp.launch(solver.update_boundary, len(bx), [s.rigid,s.boundary], device=s.device)
        r = s.boundary.local.numpy()[group==1] @ init.rotation_matrix(q).T
        np.testing.assert_allclose(s.boundary.position.numpy()[group==1], r+s.rigid.position.numpy()[1], atol=1e-7)
        np.testing.assert_allclose(s.boundary.velocity.numpy()[group==1], np.cross([0,2,0],r)+[1,2,3], atol=5e-7)
        np.testing.assert_array_equal(s.boundary.volume.numpy(), volume)

    def test_pair_impulse_balance_and_offcenter_torque(self):
        s = self.make()
        # Artificial single boundary sample isolates the sign, dt^2 and lever arm.
        b = int(np.flatnonzero(s.boundary.body.numpy()==1)[0])
        bx = s.boundary.position.numpy()[b]
        x = s.positions.numpy()
        x[0] = bx + [.04,.03,.02]
        s.positions.assign(x)
        ids = s.neighbors.boundary.numpy()
        ids[0,0] = b
        s.neighbors.boundary.assign(ids)
        count = np.zeros(s.num_particles,dtype=np.int32)
        count[0] = 1
        s.neighbors.boundary_count.assign(count)
        lambdas = np.zeros(s.num_particles,dtype=np.float32)
        lambdas[0] = -.002
        s.lambdas.assign(lambdas)
        wp.launch(solver.pressure_correction, 1,
            [s.positions,s.boundary,s.rigid,s.neighbors,s.fluid_volume,s.fluid_mass,s.support_radius,.002,1,s.lambdas,s.corrections], device=s.device)
        fluid_impulse = s.fluid_mass*s.corrections.numpy()[0]/.002
        rigid_impulse = s.rigid.force.numpy()[1]*.002
        np.testing.assert_allclose(fluid_impulse+rigid_impulse, np.zeros(3), atol=2e-5)
        self.assertGreater(np.linalg.norm(s.rigid.torque.numpy()[1]), .01)
        expected_torque = np.cross(bx-s.rigid.position.numpy()[1],s.rigid.force.numpy()[1])
        np.testing.assert_allclose(s.rigid.torque.numpy()[1], expected_torque, atol=1e-3)

    def test_static_body_and_quaternion_after_integration(self):
        s = self.make()
        initial = s.rigid.position.numpy().copy()
        force = np.array([[100,100,100],[1,0,0]],dtype=np.float32)
        s.rigid.force.assign(force)
        s.rigid.torque.assign(np.array([[100,100,100],[0,.01,0]],dtype=np.float32))
        wp.launch(solver.integrate_rigid, 2, [s.rigid,wp.vec3(0,0,0),.001], device=s.device)
        np.testing.assert_array_equal(s.rigid.position.numpy()[0],initial[0])
        self.assertGreater(s.rigid.velocity.numpy()[1,0],0)
        self.assertGreater(s.rigid.omega.numpy()[1,1],0)
        np.testing.assert_allclose(np.linalg.norm(s.rigid.rotation.numpy(),axis=1),1,atol=1e-7)

    def test_neighbor_overflow_is_reported(self):
        s = self.make(max_fluid_neighbors=1,max_boundary_neighbors=1)
        with self.assertRaisesRegex(RuntimeError,"Neighbor capacity"):
            s.step()

    def test_contact_floor_restitution_and_friction(self):
        s = self.make()
        s.contacts.count.assign(np.array([1],dtype=np.int32))
        for name,value in (("a",1),("b",0),("gap",-.001),("target",.6)):
            data = getattr(s.contacts,name).numpy()
            data[0] = value
            getattr(s.contacts,name).assign(data)
        p = s.contacts.point.numpy()
        p[0] = s.rigid.position.numpy()[1] - [0,.12,0]
        s.contacts.point.assign(p)
        normal = s.contacts.normal.numpy()
        normal[0] = [0,1,0]
        s.contacts.normal.assign(normal)
        s.rigid.friction.assign(np.array([.5,.5],dtype=np.float32))
        s.rigid.velocity.assign(np.array([[0,0,0],[1,-1,0]],dtype=np.float32))
        for _ in range(5):
            wp.launch(solver.solve_contacts,1,[s.rigid,s.contacts],device=s.device)
        self.assertAlmostEqual(s.rigid.velocity.numpy()[1,1],.6,places=5)
        self.assertLess(s.rigid.velocity.numpy()[1,0],1)
        np.testing.assert_array_equal(s.rigid.velocity.numpy()[0],[0,0,0])
        self.assertLessEqual(np.linalg.norm(s.contacts.tangent_impulse.numpy()[0]), .5*s.contacts.normal_impulse.numpy()[0]+1e-6)

    def test_mesh_contacts_and_overflow(self):
        s = self.make(max_contacts=1)
        p = s.rigid.position.numpy()
        p[1,1] = .1  # Sphere radius .12: intersects the floor.
        s.rigid.position.assign(p)
        wp.launch(solver.update_boundary,len(s.boundary.local),[s.rigid,s.boundary],device="cpu")
        wp.launch(solver.detect_contacts,(len(s.boundary.local),2),[s.rigid,s.boundary,s.contacts,.06,.002],device="cpu")
        self.assertEqual(s.contacts.overflow.numpy()[0],1)
        self.assertGreater(s.contacts.count.numpy()[0],1)

    def test_mutual_mesh_contact_transfers_momentum(self):
        data = json.loads(self.scene.read_text())
        second = dict(data["RigidBodies"][1])
        data["RigidBodies"][1]["translation"] = [-.11,.4,0]
        second["translation"] = [.11,.4,0]
        data["RigidBodies"].append(second)
        path = Path(self.temp.name)/"pair.json"
        path.write_text(json.dumps(data))
        s = self.make(scene=str(path))
        s.rigid.velocity.assign(np.array([[0,0,0],[1,0,0],[-1,0,0]],dtype=np.float32))
        wp.launch(solver.update_boundary,len(s.boundary.local),[s.rigid,s.boundary],device="cpu")
        wp.launch(solver.detect_contacts,(len(s.boundary.local),3),[s.rigid,s.boundary,s.contacts,.06,.002],device="cpu")
        self.assertGreater(s.contacts.count.numpy()[0],0)
        count = int(s.contacts.count.numpy()[0])
        self.assertTrue(np.any(s.contacts.gap.numpy()[:count]<0))
        for _ in range(5):
            wp.launch(solver.solve_contacts,1,[s.rigid,s.contacts],device="cpu")
        v = s.rigid.velocity.numpy()
        masses = np.array([b.mass for b in s.body_models])
        np.testing.assert_allclose((masses[:,None]*v).sum(axis=0),[0,0,0],atol=2e-5)
        self.assertLess(v[1,0],0)
        self.assertGreater(v[2,0],0)

    def test_cpu_cuda_short_trajectory(self):
        cpu = self.make()
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")
        gpu = solver.Example(self.config,device="cuda:0")
        for _ in range(20):
            cpu.step()
            gpu.step()
        np.testing.assert_allclose(cpu.positions.numpy(),gpu.positions.numpy(),rtol=2e-4,atol=5e-5)
        np.testing.assert_allclose(cpu.rigid.position.numpy(),gpu.rigid.position.numpy(),atol=5e-5)
        self.assertEqual(cpu.iterations,gpu.iterations)
        self.assertLessEqual(cpu.density_error_percent,cpu.config.max_density_error_percent)

    def test_invalid_configuration(self):
        for overrides in ({"particle_radius":0},{"gravity":(0,float("nan"),0)}, {"viscosity":-1},
                          {"initial_time_step":.02},{"min_iterations":101},{"max_contacts":0}):
            with self.assertRaises(ValueError):
                replace(self.config,**overrides)


if __name__ == "__main__":
    unittest.main()
