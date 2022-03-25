% --- example ---
% r = [0.2170, 0.4343, 0.8110, 1.9433, 3.1663, 4.4278];
% cdw = wavedrag(r)

function [cdw] = wavedrag(r)
    warning('off', 'MATLAB:nearlySingularMatrix');
    % Body parameters
    LENGTH = 100;
    MAX_RADIUS = 5;
    SREF = pi * MAX_RADIUS * MAX_RADIUS;
    
    % Coarse and fine mesh parameters
    NORMALIZED_X_LOCATIONS = [0, 0.005, 0.01, 0.025, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.975, 0.99, 0.995, 1.0];
    dxTiny = LENGTH / 1e4;
    dxFine = LENGTH / 1e2;
    dsTiny = LENGTH / 2e2;

    finePointCount = (LENGTH - 2 * dsTiny) / dxFine;
    tinyPointCount = 2 * (dsTiny / dxTiny);
    totalPointCount = finePointCount + tinyPointCount + 1;
    
    % Useful variables to be initialized only once
    X_LOCATIONS = zeros(15, 1);
    X_MESH = zeros(totalPointCount, 1);
    R_COARSE = zeros(15, 1);
    
    % Some radii are just fixed: nose, tail and mid-point
    R_COARSE(1) = 0;
    R_COARSE(15) = 0;
    R_COARSE(8) = MAX_RADIUS;

    % Create absolute x-locations
    for index = 1:15
        X_LOCATIONS(index) = LENGTH * NORMALIZED_X_LOCATIONS(index);
    end
    index = 1;
    % From 0 to dsTiny
    for x = 0:dxTiny:dsTiny-dxTiny
        X_MESH(index) = x;
        index = index + 1;
    end
    % From dsTiny to LENGTH-dsTiny
    for x = dsTiny:dxFine:LENGTH-dsTiny-dxFine
        X_MESH(index) = x;
        index = index + 1;
    end
    % From LENGTH-dsTiny to LENGTH
    for x = LENGTH-dsTiny:dxTiny:LENGTH-dxTiny
        X_MESH(index) = x;
        index = index + 1;
    end
    % Last point
    X_MESH(index) = LENGTH;

    % Throw exception if dimension mismatch
    if (length(r) ~= 6) 
        error("Dimension of x MUST be 6");
    end
    % Make full body
    for index = 1:6 
        R_COARSE(index + 1) = r(index);
        R_COARSE(15 - index) = r(index);
    end
    % Make fine mesh
    [u, R_FINE] = doFit(X_LOCATIONS, R_COARSE, X_MESH, 1, 1);
    % Compute areas
    for index = 1:totalPointCount
        R_FINE(index) = pi * R_FINE(index) * R_FINE(index);
    end
    pause(0.1);
    
    % Code returns cdw*sref, for some reason.
    cdw = computeDrag(X_MESH, R_FINE, totalPointCount-1) / SREF;
end

function [u, yi] = doFit(x, y, xi, ls, rs)

    n = length(x);
    ni = length(xi);
    yi = zeros(ni, 1);

    % Runs Akima spline fitting algorithm
    % Determine points to left/right of given data
    AL = zeros(3, 3);
    AR = zeros(3, 3);
    BL = zeros(3, 1);
    BR = zeros(3, 1);
    
    % Parabolic extrapolation
    AL(1, 1) = 1;
    AL(1, 2) = x(1);
    AL(1, 3) = x(1)^2;
    AL(2, 1) = 1;
    AL(2, 2) = x(2);
    AL(2, 3) = x(2)^2;
    AL(3, 1) = 1;
    AL(3, 2) = x(3);
    AL(3, 3) = x(3)^2;
    
    BL(1) = y(1);
    BL(2) = y(2);
    BL(3) = y(3);
    
    CL = AL\BL;
    
    AR(1, 1) =  1;
    AR(1, 2) =  x(n);
    AR(1, 3) =  x(n)^2;
    AR(2, 1) =  1;
    AR(2, 2) =  x(n - 1);
    AR(2, 3) =  x(n - 1)^2;
    AR(3, 1) =  1;
    AR(3, 2) =  x(n - 2);
    AR(3, 3) =  x(n - 2)^2;
    
    BR(1) = y(n);
    BR(2) = y(n - 1);
    BR(3) = y(n - 2);
    
    CR = AR\BR;
    
    % Create augmented x and y arrays
    xaug = zeros(n + 6, 1);
    xaug(1) = 3 * x(1) - 2 * x(3);
    xaug(2) = 2 * x(1) - x(3);
    xaug(3) = 2 * x(1) - x(2);
    for i = 1:n
      xaug(i + 3) = x(i);
    end
    xaug(n + 4) = 2 * x(n) - x(n - 1);
    xaug(n + 5) = 2 * x(n) - x(n - 2);
    xaug(n + 6) = 3 * x(n) - 2 * x(n - 2);
    
    % Create augmented y array
    yaug = zeros(n + 6, 1);
    for i  = 1:n
        yaug(i + 3) = y(i);
    end
    
    % Will prevent radii from being negative at the left end of the fit
    if (ls == 1)
      if (CL(2) < 0)
        CL(1) =  0.0;
        CL(2) =  y(2) / x(2);
        CL(3) =  0.0;
      end
    end
    
    yaug(1) = CL(1) + CL(2) * xaug(1) + CL(3) * (xaug(1))^2;
    yaug(2) = CL(1) + CL(2) * xaug(2) + CL(3) * (xaug(2))^2;
    yaug(3) = CL(1) + CL(2) * xaug(3) + CL(3) * (xaug(3))^2;
    

    % Will prevent radii from being negative at the right end of the fit
    if (rs == 1) 
      if (2 * CR(3) * x(n) + CR(2) > 0) 
        CR(2) =  (y(n) - y(n - 1)) / (x(n) - x(n - 1));
        CR(1) =  y(n) - CR(2) * x(n);
        CR(3) =  0;
      end
    end
    
    yaug(n + 4) = CR(1) + CR(2) * xaug(n + 4) + CR(3) * xaug(n + 4)^2;
    yaug(n + 5) = CR(1) + CR(2) * xaug(n + 5) + CR(3) * xaug(n + 5)^2;
    yaug(n + 6) = CR(1) + CR(2) * xaug(n + 6) + CR(3) * xaug(n + 6)^2;
    
    m = zeros(n + 2, 1);
    for i = 1:n+2
      m(i) = (yaug(i + 3) - yaug(i + 2)) / (xaug(i + 3) - xaug(i + 2));
    end
    % Run fitting algorithm
    
    t = zeros(n, 1);
    f = zeros(6, 1);
    v = zeros(6, 1);
    d = zeros(6, 1);
    w = zeros(6, 1);
    
    xim = zeros(4, 1);
    yim = zeros(4, 1);
    xicll = zeros(4, 1);
    yicll = zeros(4, 1);
    xicl = zeros(4, 1);
    yicl = zeros(4, 1);
    xicr = zeros(4, 1);
    yicr = zeros(4, 1);
    xicrr = zeros(4, 1);
    yicrr = zeros(4, 1);
    xip = zeros(4, 1);
    yip = zeros(4, 1);
    
    for i = 4:n+3
      sumw = 0.0;
      ii = i - 3;
      xim(1) = xaug(i);
      xim(2) = xaug(i - 2);
      xim(3) = xaug(i - 1);
      xim(4) = xaug(i - 3);
      yim(1) = yaug(i);
      yim(2) = yaug(i - 2);
      yim(3) = yaug(i - 1);
      yim(4) = yaug(i - 3);
      xicll(1) = xaug(i);
      xicll(2) = xaug(i - 1);
      xicll(3) = xaug(i + 1);
      xicll(4) = xaug(i - 3);
      yicll(1) = yaug(i);
      yicll(2) = yaug(i - 1);
      yicll(3) = yaug(i + 1);
      yicll(4) = yaug(i - 3);
      xicl(1) = xaug(i);
      xicl(2) = xaug(i - 1);
      xicl(3) = xaug(i + 1);
      xicl(4) = xaug(i - 2);
      yicl(1) = yaug(i);
      yicl(2) = yaug(i - 1);
      yicl(3) = yaug(i + 1);
      yicl(4) = yaug(i - 2);
      xicr(1) = xaug(i);
      xicr(2) = xaug(i - 1);
      xicr(3) = xaug(i + 1);
      xicr(4) = xaug(i + 2);
      yicr(1) = yaug(i);
      yicr(2) = yaug(i - 1);
      yicr(3) = yaug(i + 1);
      yicr(4) = yaug(i + 2);
      xicrr(1) = xaug(i);
      xicrr(2) = xaug(i - 1);
      xicrr(3) = xaug(i + 1);
      xicrr(4) = xaug(i + 3);
      yicrr(1) = yaug(i);
      yicrr(2) = yaug(i - 1);
      yicrr(3) = yaug(i + 1);
      yicrr(4) = yaug(i + 3);
      xip(1) = xaug(i);
      xip(2) = xaug(i + 1);
      xip(3) = xaug(i + 2);
      xip(4) = xaug(i + 3);
      yip(1) = yaug(i);
      yip(2) = yaug(i + 1);
      yip(3) = yaug(i + 2);
      yip(4) = yaug(i + 3);
      

      % Derivative estimates and curvature (volatility measure)
      fvr = Ffunc(xim, yim);
      f(1) = fvr(1);
      v(1) = fvr(2);
      fvr = Ffunc(xicll, yicll);
      f(2) = fvr(1);
      v(2) = fvr(2);
      fvr = Ffunc(xicl, yicl);
      f(3) = fvr(1);
      v(3) = fvr(2);
      fvr = Ffunc(xicr, yicr);
      f(4) = fvr(1);
      v(4) = fvr(2);
      fvr = Ffunc(xicrr, yicrr);
      f(5) = fvr(1);
      v(5) = fvr(2);
      fvr = Ffunc(xip, yip);
      f(6) = fvr(1);
      v(6) = fvr(2);
      
      % Distance measures
      d(1) = Dfunc(xim);
      d(2) = Dfunc(xicll);
      d(3) = Dfunc(xicl);
      d(4) = Dfunc(xicr);
      d(5) = Dfunc(xicrr);
      d(6) = Dfunc(xip);
      for j = 1:6 
        v(j) = v(j) + 1.0e-200; % in case some of the v's are zero
        w(j) = 1 / v(j);
      end
      
      wmax = max(w);
      
      for j = 1:6
        w(j) = w(j) / (wmax * d(j)); % renormalize to get rid of 1e+200
        t(ii) = t(ii) + f(j) * w(j);
        sumw = sumw + w(j);
      end
      t(ii) = t(ii) / sumw;
      
    end
    
    u = zeros(4, ni);
    for j = 1:ni
      ii = 2;
      while ((xi(j) + 1e-6) > x(ii))
        ii = ii + 1;
        if (xi(j) >= x(n)) 
          ii = n;
          break;
        end
      end
      i = ii - 1;
      
      hk = x(i + 1) - x(i);
      p0 = y(i);
      p1 = t(i);
      p2 = (3 * m(ii) - 2 * t(i) - t(i + 1)) / hk;
      p3 = (t(i) + t(i + 1) - 2 * m(ii)) / hk^2;
      
      % Fit value!!!
      hi = xi(j) - x(i);
      yi(j) = p0 + p1 * hi + p2 * hi * hi + p3 * hi * hi * hi;
      % Compute the coefficients of the cubic fit
      u(1, j) = p0 - p1 * x(i) + p2 * x(i)^2 - p3 * x(i)^3;
      u(2, j) = p1 - 2 * p2 * x(i) + 3 * p3 * x(i)^2;
      u(3, j) = p2 - 3 * p3 * x(i);
      u(4, j) = p3;
      

    end
    
end

function [fv] = Ffunc(x, y)  
    
    % Initialize needed matrices
    A = zeros(4, 4);
    B = zeros(4, 1);
    h = zeros(2, 1);
    
    A(1, 1) = 1;
    A(1, 2) = x(1);
    A(1, 3) = x(1)^2;
    A(1, 4) = x(1)^3;
    A(2, 1) = 1;
    A(2, 2) = x(2);
    A(2, 3) = x(2)^2;
    A(2, 4) = x(2)^3;
    A(3, 1) = 1;
    A(3, 2) = x(3);
    A(3, 3) = x(3)^2;
    A(3, 4) = x(3)^3;
    A(4, 1) = 1;
    A(4, 2) = x(4);
    A(4, 3) = x(4)^2;
    A(4, 4) = x(4)^3;
    
    B(1) = y(1);
    B(2) = y(2);
    B(3) = y(3);
    B(4) = y(4);
    
    C = A\B;
    
    f = C(2) + 2 * C(3) * x(1) + 3 * C(4) * x(1)^2;
    mx = min(x);
    Mx = max(x);
    h(1) = (2 * C(3) + 6 * C(4) * mx)^2;
    h(2) = (2 * C(3) + 6 * C(4) * Mx)^2;
    v = max(h);
    fv = zeros(2, 1);
    fv(1) = f;
    fv(2) = v;
end
  

function [d] = Dfunc(x)
    d = (x(2) - x(1))^2 + (x(3) - x(1))^2 + (x(4) - x(1))^2;
end




function [cdwsref] = computeDrag(x, A, nmax)
    rs = 1;
    if (abs(A(nmax)) > 1 * 10^-10) 
      rs = 0;
    end
%     Run Akima code to get fit
    [u, yi] = doFit(x, A, x, 1, rs);
    
    % Compute double integral
    I = 0;
    
    for i = 1:nmax
      x1l = x(i);
      x1u = x(i + 1);
      g0 = 2 * u(3, i);
      g1 = 6 * u(4, i);
      dI = 0;
      
      for j = 1:i 
        x2l = x(j);
        x2u = x(j + 1);
        c0 = 2 * u(3, j);
        c1 = 6 * u(4, j);
        
        t1pm = -(c0 / 6) * (3 * g0 + 2 * g1 * (x1u - x2l) + 3 * g1 * x2l) ...
                * (x1u - x2l)^2;
        t2pm = (c1 / 48) * (4 * g0 + 3 * g1 * (x1u - x2l) + 4 * g1 * x2l) ...
                * (x1u - x2l)^3;
        t3pm = -(c1 / 12) ...
                * (4 * g0 * (x1u - x2l) + 3 * g1 * (x1u - x2l)^2 + 6 ...
                    * g0 * x2l + 8 * g1 * (x1u - x2l) * x2l + 6 * g1 ...
                    * x2l^2) * (x1u - x2l)^2;
        t4pm = (c0 / 36) ...
                * (-9 * g0 - 4 * g1 * (x1u - x2l) - 9 * g1 * x2l + 6 ...
                    * (3 * g0 + 2 * g1 * (x1u - x2l) + 3 * g1 * x2l) ...
                    * log(x1u - x2l)) * (x1u - x2l)^2;
        t5pm = -(c1 / 288) ...
                * (-16 * g0 - 9 * g1 * (x1u - x2l) - 16 * g1 * x2l + 12 ...
                    * (4 * g0 + 3 * g1 * (x1u - x2l) + 4 * g1 * x2l) ...
                    * log(x1u - x2l)) * (x1u - x2l)^3;
        t6pm = (c1 / 144) * (-4 * g0 * (4 * (x1u - x2l) + 9 * x2l) ...
                    - g1 * (9 * (x1u - x2l)^2 + 32 * (x1u - x2l) * x2l + 36 * ...
                        x2l^2) + 12 * (4 * g0 * (x1u - x2l) + 3 * g1 * (x1u - x2l)^2 ...
                        + 6 * g0 * x2l + 8 * g1 * (x1u - x2l) * x2l + 6 * g1 ...
                        * x2l^2) * log(x1u - x2l)) ...
                * (x1u - x2l)^2;
        
        if (i == j)
          t1pp = 0;
          t1mm = 0;
          t1mp = 0;
          t2pp = 0;
          t2mm = 0;
          t2mp = 0;
          t3pp = 0;
          t3mm = 0;
          t3mp = 0;
          t4pp = 0;
          t4mm = 0;
          t4mp = 0;
          t5pp = 0;
          t5mm = 0;
          t5mp = 0;
          t6pp = 0;
          t6mm = 0;
          t6mp = 0;
        else
          t1pp = (c0 / 6) * (3 * g0 + 2 * g1 * (x1u - x2u) + 3 * g1 * x2u) ...
                  * (x1u - x2u)^2;
          t1mm = (c0 / 6) * (3 * g0 + 2 * g1 * (x1l - x2l) + 3 * g1 * x2l) ...
                  * (x1l - x2l)^2;          
          t2pp = -(c1 / 48) * (4 * g0 + 3 * g1 * (x1u - x2u) + 4 * g1 * x2u) ...
                  * (x1u - x2u)^3;
          t2mm = -(c1 / 48) * (4 * g0 + 3 * g1 * (x1l - x2l) + 4 * g1 * x2l) ...
                  * (x1l - x2l)^3;
          t3pp = (c1 / 12) ...
                  * (4 * g0 * (x1u - x2u) + 3 * g1 * (x1u - x2u)^2 + 6 ...
                      * g0 * x2u + 8 * g1 * (x1u - x2u) * x2u + 6 * g1 ...
                      * (x2u)^2) * (x1u - x2u)^2;
          t3mm = (c1 / 12) ...
                  * (4 * g0 * (x1l - x2l) + 3 * g1 * (x1l - x2l)^2 + 6  ...
                      * g0 * x2l + 8 * g1 * (x1l - x2l) * x2l + 6 * g1 ...
                      * (x2l)^2) * (x1l - x2l)^2;
          t4pp = -(c0 / 36) ...
                  * (-9 * g0 - 4 * g1 * (x1u - x2u) - 9 * g1 * x2u + 6 ...
                      * (3 * g0 + 2 * g1 * (x1u - x2u) + 3 * g1 * x2u) ...
                      * log(x1u - x2u)) * (x1u - x2u)^2;
          t4mm = -(c0 / 36) ...
                  * (-9 * g0 - 4 * g1 * (x1l - x2l) - 9 * g1 * x2l + 6 ...
                      * (3 * g0 + 2 * g1 * (x1l - x2l) + 3 * g1 * x2l) ...
                      * log(x1l - x2l)) * (x1l - x2l)^2;
          t5pp = (c1 / 288) ...
                  * (-16 * g0 - 9 * g1 * (x1u - x2u) - 16 * g1 * x2u + 12 ...
                      * (4 * g0 + 3 * g1 * (x1u - x2u) + 4 * g1 * x2u) ...
                      * log(x1u - x2u)) * (x1u - x2u)^3;
          t5mm = (c1 / 288) ...
                  * (-16 * g0 - 9 * g1 * (x1l - x2l) - 16 * g1 * x2l + 12 ...
                      * (4 * g0 + 3 * g1 * (x1l - x2l) + 4 * g1 * x2l) ...
                      * log(x1l - x2l)) * (x1l - x2l)^3;
          t6pp = -(c1 / 144) * (-4 * g0 ...
                      * (4 * (x1u - x2u) + 9 * x2u) - g1 ...
                      * (9 * (x1u - x2u)^2 + 32 * (x1u - x2u) * x2u + 36 * ...
                          (x2u)^2) + 12 ...
                      * (4 * g0 * (x1u - x2u) + 3 * g1 * (x1u - x2u)^2 ...
                          + 6 * g0 * x2u + 8 * g1 * (x1u - x2u) * x2u + 6 * g1 ...
                          * (x2u)^2) * log(x1u - x2u)) * (x1u - x2u)^2;
          t6mm = -(c1 / 144) * (-4 * g0 ...
                      * (4 * (x1l - x2l) + 9 * x2l) - g1 ...
                      * (9 * (x1l - x2l)^2 + 32 * (x1l - x2l) * x2l + 36 * (x2l)^2) + 12 ...
                      * (4 * g0 * (x1l - x2l) + 3 * g1 * (x1l - x2l)^2 ...
                          + 6 * g0 * x2l + 8 * g1 * (x1l - x2l) * x2l + 6 * g1 ...
                          * (x2l)^2) * log(x1l - x2l)) * (x1l - x2l)^2;
          
          if (j == i - 1)
            t1mp = 0;
            t2mp = 0;
            t3mp = 0;
            t4mp = 0;
            t5mp = 0;
            t6mp = 0;
          else 
            t1mp = -(c0 / 6) * (3 * g0 + 2 * g1 * (x1l - x2u) + 3 * g1 * x2u) ...
                    * (x1l - x2u)^2;
            t2mp = (c1 / 48) * (4 * g0 + 3 * g1 * (x1l - x2u) + 4 * g1 * x2u) ...
                    * (x1l - x2u)^3;
            t3mp = -(c1 / 12) ...
                    * (4 * g0 * (x1l - x2u) + 3 * g1 * (x1l - x2u)^2 ...
                        + 6 * g0 * x2u + 8 * g1 * (x1l - x2u) * x2u + 6 * g1 ...
                        * (x2u)^2) * (x1l - x2u)^2;
            t4mp = (c0 / 36) ...
                    * (-9 * g0 - 4 * g1 * (x1l - x2u) - 9 * g1 * x2u + 6 ...
                        * (3 * g0 + 2 * g1 * (x1l - x2u) + 3 * g1 * x2u) ...
                        * log(x1l - x2u)) * (x1l - x2u)^2;
            t5mp = -(c1 / 288) ...
                    * (-16 * g0 - 9 * g1 * (x1l - x2u) - 16 * g1 * x2u + 12 ...
                        * (4 * g0 + 3 * g1 * (x1l - x2u) + 4 * g1 * x2u) ...
                        * log(x1l - x2u)) * (x1l - x2u)^3;
            t6mp = (c1 / 144) * (-4 * g0 ...
                        * (4 * (x1l - x2u) + 9 * x2u) - g1 ...
                        * (9 * (x1l - x2u)^2 + 32 * (x1l - x2u) * x2u + 36 * (x2u)^2) + 12 ...
                        * (4 * g0 * (x1l - x2u) + 3 * g1 ...
                            * (x1l - x2u)^2 + 6 * g0 * x2u + 8 * g1 ...
                            * (x1l - x2u) * x2u + 6 * g1 * (x2u)^2) ...
                        * log(x1l - x2u)) * (x1l - x2u)^2;
          end
          
        end
        
        t1 = t1pp + t1pm + t1mp + t1mm;
        t2 = t2pp + t2pm + t2mp + t2mm;
        t3 = t3pp + t3pm + t3mp + t3mm;
        t4 = t4pp + t4pm + t4mp + t4mm;
        t5 = t5pp + t5pm + t5mp + t5mm;
        t6 = t6pp + t6pm + t6mp + t6mm;
        
        dI = dI + t1 + t2 + t3 + t4 + t5 + t6;
      end
      
      I = I + dI;
      
    end
    
    cdwsref = -2 * I / (2 * pi);
    
end

 
% function [sref] = computeSref(x, A, nx, L)
%     sref = 0;
%     len = nx;
%     
%     rs = 1; % If rs=1 forces radii of fit to be >0.
%     if (abs(A(len)) > 10^-1)
%         rs = 0;
%     end
%     
%     % Find max value of input areas
%     iAmax = 0;
%     sref = 0;
%     for i = 1:len+1
%         if (A(i) > sref)
%             sref = A(i);
%             iAmax = i;
%         end
%     end
%     
%     % Get Coefficients of Akima spline fit to radius distribution
%     xavg = zeros(len-1, 1);
%     for i = 1:len-1
%       xavg(i) = .5 * (x(i + 1) + x(i));
%     end
%     
%     % Run Akima code to get fit
%     [u, yi] = doFit(x, A, xavg, 1, rs);
%     % Split u into arrays holding coefficients of 3rd order fit to area
%     % distribution
%     a = zeros(len - 1, 1);
%     b = zeros(len - 1, 1);
%     c = zeros(len - 1, 1);
%     d = zeros(len - 1, 1);
%     for i = 1:len
%       a(i) = u(1, i);
%       b(i) = u(2, i);
%       c(i) = u(3, i);
%       d(i) = u(4, i);
%     end
%     
%     % Max area can be at this point or to the left or right of it. Check slope
%     % to see
%     % which direction it may be.
%     if (3 * a(iAmax - 1) * x(iAmax)^2 + 2 * b(iAmax - 1) * x(iAmax) + c(iAmax - 1) < 0.0) % Look left
%     
%       % Find where derivative is 0 (roots of quadratic)
%       x1 = (-2 * b(iAmax - 1) - sqrt(4 * b(iAmax - 1)^2 - 12 ...
%               * a(iAmax - 1) * c(iAmax - 1))) ...
%               / (6 * a(iAmax - 1));
%       x2 = (-2 * b(iAmax - 1) + sqrt(4 * b(iAmax - 1)^2 - 12 ...
%               * a(iAmax - 1) * c(iAmax - 1))) ...
%               / (6 * a(iAmax - 1));
%       
%       if ((x1 <= x(iAmax - 1)) && (x1 <= x(iAmax))) 
%         sref = a(iAmax - 1) * x1^3 + b(iAmax - 1) * x1^2 ...
%                 + c(iAmax - 1) * x1 + d(iAmax - 1);
%       else
%         sref = a(iAmax - 1) * x2^3 + b(iAmax - 1) * x2^2 ...
%                 + c(iAmax - 1) * x2 + d(iAmax - 1);
%       end
%       
%     elseif (3 * a(iAmax) * x(iAmax)^2 + 2 * b(iAmax) * x(iAmax) + c(iAmax) > 0.0) % Look right
%     
%       x1 = (-2 * b(iAmax) - Math.sqrt(4 * Math.pow(b(iAmax), 2) - 12 * a(iAmax) ...
%               * c(iAmax))) / (6 * a(iAmax));
%       double x2 = (-2 * b(iAmax) + Math.sqrt(4 * Math.pow(b(iAmax), 2) - 12 * a(iAmax) ...
%               * c(iAmax))) / (6 * a(iAmax));
%       
%       if ((x1 >= x(iAmax)) && (x1 <= x(iAmax + 1))) 
%         sref = a(iAmax) * Math.pow(x1, 3) + b(iAmax) * Math.pow(x1, 2) + c(iAmax) * x1 + d(iAmax);
%       else
%         sref = a(iAmax) * Math.pow(x2, 3) + b(iAmax) * Math.pow(x2, 2) + c(iAmax) * x2 + d(iAmax);
%       end
%       
%     end
% end
  