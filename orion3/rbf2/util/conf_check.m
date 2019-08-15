function conf = conf_check(conf, spec, prog)
%
% Check the legality of a configuration against its specification.
%
% Checks configuration parameters (in the form of a structure)
% against the specifications they are meant to conform to.
% Fills in any defaults for fields which are present in the
% specification but missing from the structure.
%

if isempty(conf)

  % Just fill in all the defaults.
  for i = 1:length(spec)
    eval(['conf.' spec(i).name ' = spec(i).default;']);
  end

elseif isstruct(conf)

  % Warn about inappropriate fields.
  allowed = {spec.name};
  actual = fieldnames(conf);
  for i = 1:length(actual)
    if isempty(find(strcmp(actual{i}, allowed)))
      fprintf('%s: (warning) unrecognised configuration field ''%s''\n', prog, actual{i})
      conf = rmfield(conf, actual{i});
    end
  end

  % Check types and options and fill in missing fields.
  for i = 1:length(spec)

    if isfield(conf, spec(i).name)
      obj = getfield(conf, spec(i).name);

      % Check type.
      if iscell(spec(i).type)
        types = spec(i).type;
      else
        types = {spec(i).type};
      end
      for t = 1:length(types)
        okay = 0;
        switch types{t}
        case 'structure'
          okay = isstruct(obj);
        case 'string'
          okay = ischar(obj) & ndims(obj) == 2 & (size(obj,1) == 1 | isempty(obj));
        case 'number'
          okay = isnumeric(obj) & ndims(obj) == 2 & max(size(obj)) == 1;
        case 'vector'
          okay = isnumeric(obj) & ndims(obj) == 2 & min(size(obj)) == 1;
        case 'row vector'
          okay = isnumeric(obj) & ndims(obj) == 2 & size(obj,1) == 1;
        case {'col vector', 'column vector'}
          okay = isnumeric(obj) & ndims(obj) == 2 & size(obj,2) == 1;
        case 'matrix'
          okay = isnumeric(obj) & ndims(obj) == 2;
        case 'length 2'
          okay = length(obj) == 2;
        case 'integer'
          okay = isempty(find(obj ~= round(obj)));
        case 'positive'
          okay = isempty(find(obj <= 0));
        case {'nonnegative', 'non-negative'}
          okay = isempty(find(obj < 0));
        case 'negative'
          okay = isempty(find(obj >= 0));
        case {'nonpositive', 'non-positive'}
          okay = isempty(find(obj > 0));
        case 'even'
          okay = isempty(find(rem(obj,2) ~= 0));
        case 'odd'
          okay = isempty(find(rem(obj,2) ~= 1));
        case 'cellstr'
          okay = iscellstr(obj);
        otherwise
          error(['conf_check: field type ''' types{t} ''' unknown'])
        end
        if ~okay
          error([prog ': configuration field ''' spec(i).name ''' does not match type ''' types{t} ''''])
        end
      end

      % Check object matches options (if specified).
      opts = spec(i).options;
      if ~isempty(opts)
        if ~iscell(opts)
          opts = {opts};
        end
        omatch = 0;
        for o = 1:length(opts)
          alts = opts{o};
          if ~iscell(alts)
            alts = {alts};
          end
          amatch = 0;
          for a = 1:length(alts)
            if isnumeric(obj)
              amatch = (alts{a} == obj);
            else
              amatch = strcmp(alts{a}, lower(obj));
            end
            if amatch
              break;
            end
          end
          if amatch
            eval(['conf.' spec(i).name ' = alts{1};']);
            omatch = 1;
            break;
          end
        end
        if ~omatch
          error([prog ': contents of field ''' spec(i).name ''' do not match any options'])
        end
      end

    else

      % Set default.
      eval(['conf.' spec(i).name ' = spec(i).default;'])

    end

  end

else

  % Shouldn't happen.
  error([prog ': configuration parameter should be struct'])

end
