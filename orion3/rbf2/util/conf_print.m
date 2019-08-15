function conf_print(prog, specs, field)
%
% Print a configuration specification, or one part of it.
%
% Prints out a configuration specification to remind the user what
% fields are allowed, what their type is, what the options are and
% what defaults are used. If the third argument is present, only
% information about that named field (if it exists) is printed.
%
 
% Print header (and check for presence of field if third arg sent).
if nargin == 2
  title = [prog ': configuration specification'];
  fprintf('%s\n', title)
  dash = '-';
  fprintf('%s\n', dash(1,ones(1,length(title))))
else
  k = find(strcmp({specs.name}, field));
  if length(k) == 1
    specs = specs(k);
    title = [prog ': configuration field ''' field ''''];
    fprintf('%s\n', title)
  elseif length(k) == 0
    fprintf('%s is not a valid configuration field for %s\n', field, prog)
    return
  else
    fprintf('%s: (program error!!) two fields have the same name (%s)\n', prog, field)
    return
  end
end

% One section for each field.
for i = 1:length(specs)
  spec = specs(i);
  
  % Comment.
  fprintf('%s:\n', spec.comment)

  % Field name.
  fprintf('    field: %s\n', spec.name)

  % Type restriction(s).
  if ischar(spec.type)
    fprintf('     type: %s\n', spec.type)
  else
    fprintf('    types:')
    for j = 1:length(spec.type)
      if j > 1
        fprintf(',')
      end
      fprintf(' %s', spec.type{j})
    end
    fprintf('\n')
  end

  % Permitted options.
  if isempty(spec.options)
    fprintf('  options: (none specified)\n')
  else
    opts = spec.options;
    if ~iscell(opts)
      opts = {opts};
    end
    fprintf('  options:')
    for o = 1:length(opts)
      if o > 1
        fprintf(',')
      end
      fprintf(' ')
      alts = opts{o};
      if ~iscell(alts)
        alts = {alts};
      end
      for a = 1:length(alts)
        if a > 1
          fprintf('|')
        end
        if isnumeric(alts{a})
          fprintf('%s', num2str(alts{a}))
        else
          fprintf('''%s''', alts{a})
        end
      end
    end
    fprintf('\n')
  end

  % Default (assume empty, string or numeric scalar/vector).
  fprintf('  default: ')
  if isempty(spec.default)
    fprintf('[]')
  elseif ischar(spec.default)
    fprintf('''%s''', spec.default)
  else
    if length(spec.default) > 1
      fprintf('[')
    end
    if isempty(find(round(spec.default) ~= spec.default))
      format = '%d';
    else
      if max(abs(spec.default)) < 100 & min(abs(spec.default)) > 0.01
        format = '%.2f';
      else
        format = '%.2e';
      end
    end
    for l = 1:length(spec.default)
      fprintf(format, spec.default(l))
      if l < length(spec.default)
        fprintf(' ')
      end
    end
    if length(spec.default) > 1
      fprintf(']')
    end
  end
  fprintf('\n')

end
fprintf('\n')
